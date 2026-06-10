"use server";

import { auth } from "@/lib/auth";
import { prisma } from "@/lib/prisma";
import { revalidatePath } from "next/cache";
import { headers } from "next/headers";
import { redirect } from "next/navigation";

export async function createPlaylist(name: string) {
  const session = await auth.api.getSession({ headers: await headers() });
  if (!session) redirect("/auth/sign-in");

  const playlist = await prisma.playlist.create({
    data: { name, userId: session.user.id },
  });

  revalidatePath("/");
  return playlist;
}

export async function getUserPlaylists() {
  const session = await auth.api.getSession({ headers: await headers() });
  if (!session) redirect("/auth/sign-in");

  return prisma.playlist.findMany({
    where: { userId: session.user.id },
    include: {
      songs: {
        include: {
          song: {
            select: { id: true, thumbnailS3Key: true, title: true },
          },
        },
        orderBy: { addedAt: "asc" },
        take: 4,
      },
      _count: { select: { songs: true } },
    },
    orderBy: { createdAt: "desc" },
  });
}

export async function getPlaylistById(playlistId: string) {
  const session = await auth.api.getSession({ headers: await headers() });
  if (!session) redirect("/auth/sign-in");

  return prisma.playlist.findUnique({
    where: { id: playlistId, userId: session.user.id },
    include: {
      songs: {
        include: {
          song: {
            include: {
              user: { select: { name: true } },
              _count: { select: { likes: true } },
              likes: { where: { userId: session.user.id } },
            },
          },
        },
        orderBy: { addedAt: "asc" },
      },
    },
  });
}

export async function addSongToPlaylist(playlistId: string, songId: string) {
  const session = await auth.api.getSession({ headers: await headers() });
  if (!session) redirect("/auth/sign-in");

  await prisma.playlistSong.upsert({
    where: { playlistId_songId: { playlistId, songId } },
    update: {},
    create: { playlistId, songId },
  });

  revalidatePath("/");
  revalidatePath(`/playlist/${playlistId}`);
}

export async function removeSongFromPlaylist(playlistId: string, songId: string) {
  const session = await auth.api.getSession({ headers: await headers() });
  if (!session) redirect("/auth/sign-in");

  await prisma.playlistSong.delete({
    where: { playlistId_songId: { playlistId, songId } },
  });

  revalidatePath("/");
  revalidatePath(`/playlist/${playlistId}`);
}

export async function deletePlaylist(playlistId: string) {
  const session = await auth.api.getSession({ headers: await headers() });
  if (!session) redirect("/auth/sign-in");

  await prisma.playlist.delete({
    where: { id: playlistId, userId: session.user.id },
  });

  revalidatePath("/");
}

export async function renamePlaylist(playlistId: string, name: string) {
  const session = await auth.api.getSession({ headers: await headers() });
  if (!session) redirect("/auth/sign-in");

  await prisma.playlist.update({
    where: { id: playlistId, userId: session.user.id },
    data: { name },
  });

  revalidatePath("/");
}

export async function getLikedSongsPlaylist() {
  const session = await auth.api.getSession({ headers: await headers() });
  if (!session) redirect("/auth/sign-in");

  // find or create the special "Liked Songs" playlist
  let playlist = await prisma.playlist.findFirst({
    where: { userId: session.user.id, name: "Liked Songs" },
    include: {
      songs: {
        include: {
          song: {
            include: {
              user: { select: { name: true } },
              _count: { select: { likes: true } },
              likes: { where: { userId: session.user.id } },
            },
          },
        },
        orderBy: { addedAt: "asc" },
      },
    },
  });

  if (!playlist) {
    playlist = await prisma.playlist.create({
      data: { name: "Liked Songs", userId: session.user.id },
      include: {
        songs: {
          include: {
            song: {
              include: {
                user: { select: { name: true } },
                _count: { select: { likes: true } },
                likes: { where: { userId: session.user.id } },
              },
            },
          },
          orderBy: { addedAt: "asc" },
        },
      },
    });
  }

  return playlist;
}