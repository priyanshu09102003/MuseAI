
import { prisma } from "@/lib/prisma";
import { inngest } from "./client";


export const generateSong = inngest.createFunction(
  { id: "generate-song", triggers: { event: "song-generation" } },
  async ({ event, step }) => {
    const {songId} = event.data as {
        songId: string;
        userId: string;
    };

    await step.run("check-credits", async () => {
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
        
        let endpoint : string | undefined
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
    })
  }
);