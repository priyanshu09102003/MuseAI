"use server"

import { inngest } from "@/inngest/client";
import { auth } from "@/lib/auth"
import { prisma } from "@/lib/prisma"
import { headers } from "next/headers"
import { redirect } from "next/navigation";

export async function queueSong(){
    const session = await auth.api.getSession({
        headers: await headers(),
    }); 

    if (!session) redirect("/auth/sign-in");

    const song = await prisma.song.create({
        data:{
            userId: session.user.id,
            title: "Test song 1",
            fullDescribedSong: "Hip-hop song"
        }
    });

    await inngest.send({
        name: "generate-song-event",
        data:{
            songId: song.id,
            userId: song.userId
        }
    })
}