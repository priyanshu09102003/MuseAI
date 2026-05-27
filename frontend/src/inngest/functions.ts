
import { prisma } from "@/lib/prisma";
import { inngest } from "./client";



export const generateSong = inngest.createFunction(
  { id: "generate-song", concurrency:{
    limit: 1,
    key: "event.data.userId"
  },

  onFailure: async({event, error}) => {
    await prisma.song.update({
        where:{
            id: (event?.data?.event?.data as { songId: string }).songId,
        },
        data: {
          status: "failed",
        },
    });
  },
   
  triggers: { event: "generate-song-event" } },
  async ({ event, step }) => {
    const {songId} = event.data as {
        songId: string;
        userId: string;
    };

    const {userId, credits, endpoint, body} = await step.run("check-credits", async () => {
        const song = await prisma.song.findFirstOrThrow({
            where:{
                id: songId
            },
            select:{
                user:{
                    select:{
                        id: true,
                        credits: true
                    }
                },
                prompt: true,
                lyrics: true,
                fullDescribedSong: true,
                describedLyrics: true,
                instrumental: true,
                guidanceScale: true,
                inferStep: true,
                audioDuration: true,
                seed: true
            }
        });

        type RequestBody = {
            guidance_scale?: number;
            infer_step?: number;
            audio_duration?: number;
            seed?: number;
            full_described_song?: string;
            prompt?: string;
            lyrics?: string;
            described_lyrics?: string;
            instrumental?: boolean;
        }
        
        let endpoint: string | undefined
        let body: RequestBody = {}

        const commomParams = {
          guidance_scale: song.guidanceScale ?? undefined,
          infer_step: song.inferStep ?? undefined,
          audio_duration: song.audioDuration ?? undefined,
          seed: song.seed ?? undefined,
          instrumental: song.instrumental ?? undefined,
        };


        // CASE I : User provides the Description of a song
        if (song.fullDescribedSong) {
          endpoint = process.env.GENERATE_FROM_DESCRIPTION;
          body = {
            full_described_song: song.fullDescribedSong,
            ...commomParams,
          };
        }

        // CASE II : Custom mode: Lyrics + prompt
        else if (song.lyrics && song.prompt) {
          endpoint = process.env.GENERATE_WITH_LYRICS;
          body = {
            lyrics: song.lyrics,
            prompt: song.prompt,
            ...commomParams,
          };
        }

        // CASE III : Custom mode: Prompt + described lyrics
        else if (song.describedLyrics && song.prompt) {
          endpoint = process.env.GENERATE_FROM_DESCRIBED_LYRICS;
          body = {
            described_lyrics: song.describedLyrics,
            prompt: song.prompt,
            ...commomParams,
          };
        }

        return {
          userId: song.user.id,
          credits: song.user.credits,
          endpoint: endpoint,
          body: body,
        };
    });

    if(credits>0){
        if (!endpoint) {
            throw new Error("No valid song generation case matched — endpoint is undefined");
        }

        await step.run("set-status-processing", async () => {
        return await prisma.song.update({
          where: {
            id: songId,
          },
          data: {
            status: "processing",
          },
        });
      });

      const response = await step.fetch(endpoint, {
        method: "POST",
        body: JSON.stringify(body),
        headers: {
          "Content-Type": "application/json",
          "Modal-Key": process.env.MODAL_KEY ?? "",
          "Modal-Secret": process.env.MODAL_SECRET ?? "",
        },
      });


      await step.run("update-song-result", async () => {
        const responseData = response.ok
          ? ((await response.json()) as {
              s3_key: string;
              cover_image_s3_key: string;
              categories: string[];
            })
          : null;

        await prisma.song.update({
          where: {
            id: songId,
          },
          data: {
            s3Key: responseData?.s3_key,
            thumbnailS3Key: responseData?.cover_image_s3_key,
            status: response.ok ? "processed" : "failed",
          },
        });

        if (responseData && responseData.categories.length > 0) {
          await prisma.song.update({
            where: { id: songId },
            data: {
              categories: {
                connectOrCreate: responseData.categories.map(
                  (categoryName) => ({
                    where: { name: categoryName },
                    create: { name: categoryName },
                  }),
                ),
              },
            },
          });
        }
      });

      return await step.run("deduct-credits", async () => {
        if (!response.ok) return;

        return await prisma.user.update({
          where: { id: userId },
          data: {
            credits: {
              decrement: 1,
            },
          },
        });
      });

    }

    else{
        //Set status: Not enough credits
        await step.run("set-status-no-credits", async () => {
        return await prisma.song.update({
          where: {
            id: songId,
          },
          data: {
            status: "no credits",
          },
        });
      });
    }
  }
);