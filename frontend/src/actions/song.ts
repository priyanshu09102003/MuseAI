"use server";

import { auth } from "@/lib/auth";
import { prisma } from "@/lib/prisma";
import { revalidatePath } from "next/cache";
import { headers } from "next/headers";
import { redirect } from "next/navigation";



export async function setPublishedStatus(songId: string, published: boolean) {
  const session = await auth.api.getSession({
    headers: await headers(),
  });

  if (!session) redirect("/auth/sign-in");

  await prisma.song.update({
    where: {
      id: songId,
      userId: session.user.id,
    },
    data: {
      published,
    },
  });

  revalidatePath("/create");
}

export async function renameSong(songId: string, newTitle: string) {
  const session = await auth.api.getSession({
    headers: await headers(),
  });

  if (!session) redirect("/auth/sign-in");

  await prisma.song.update({
    where: {
      id: songId,
      userId: session.user.id,
    },
    data: {
      title: newTitle,
    },
  });

  revalidatePath("/create");
}

export async function toggleLikeSong(songId: string) {
  const session = await auth.api.getSession({ headers: await headers() });
  if (!session) redirect("/auth/sign-in");

  const existingLike = await prisma.like.findUnique({
    where: { userId_songId: { userId: session.user.id, songId } },
  });

  // find or create Liked Songs playlist
  let likedPlaylist = await prisma.playlist.findFirst({
    where: { userId: session.user.id, name: "Liked Songs" },
  });
  if (!likedPlaylist) {
    likedPlaylist = await prisma.playlist.create({
      data: { name: "Liked Songs", userId: session.user.id },
    });
  }

  if (existingLike) {
    await prisma.like.delete({
      where: { userId_songId: { userId: session.user.id, songId } },
    });
    await prisma.playlistSong.deleteMany({
      where: { playlistId: likedPlaylist.id, songId },
    });
  } else {
    await prisma.like.create({ data: { userId: session.user.id, songId } });
    await prisma.playlistSong.upsert({
      where: { playlistId_songId: { playlistId: likedPlaylist.id, songId } },
      update: {},
      create: { playlistId: likedPlaylist.id, songId },
    });
  }

  revalidatePath("/");
}