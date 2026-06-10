import { getPresignedUrl } from "@/actions/generation";
import { getPlaylistById } from "@/actions/playlist";
import { PlaylistHero } from "@/custom_components/playlist/playlist-hero";
import { auth } from "@/lib/auth";
import { headers } from "next/headers";
import { redirect, notFound } from "next/navigation";
import { Music } from "lucide-react";


export default async function PlaylistPage({ params }: { params: Promise<{ id: string }> }) {
  const session = await auth.api.getSession({ headers: await headers() });
  if (!session) redirect("/auth/sign-in");

  const { id } = await params;
  const playlist = await getPlaylistById(id);
  if (!playlist) notFound();

  const songsWithUrls = await Promise.all(
    playlist!.songs.map(async (ps) => {
      const thumbnailUrl = ps.song.thumbnailS3Key
        ? await getPresignedUrl(ps.song.thumbnailS3Key)
        : null;
      return { ...ps.song, thumbnailUrl };
    })
  );

  const totalDuration = songsWithUrls.reduce(
    (acc, s) => acc + (s.audioDuration ?? 0), 0
  );
  const hours = Math.floor(totalDuration / 3600);
  const mins = Math.floor((totalDuration % 3600) / 60);
  const durationStr = hours > 0 ? `${hours} hr ${mins} min` : `${mins} min`;

  return (
    <div className="flex flex-col min-h-full">
      <PlaylistHero
        playlist={playlist!}
        songs={songsWithUrls}
        songCount={songsWithUrls.length}
        durationStr={durationStr}
      />
    </div>
  );
}