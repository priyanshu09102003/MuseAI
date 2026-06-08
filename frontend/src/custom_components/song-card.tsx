"use client";

import { Heart, Loader2, Music, Play } from "lucide-react";
import { useState } from "react";
import { getPlayUrl } from "@/actions/generation";
import { toggleLikeSong } from "@/actions/song";
import { usePlayerStore } from "@/stores/use-player";
import { Category, Like, Song } from "@/generated/prisma";

type SongWithRelation = Song & {
  user: { name: string | null };
  _count: { likes: number };
  categories: Category[];
  thumbnailUrl?: string | null;
  likes?: Like[];
};

export function SongCard({ song }: { song: SongWithRelation }) {
  const [isLoading, setIsLoading] = useState(false);
  const setTrack = usePlayerStore((state) => state.setTrack);
  const [isLiked, setIsLiked] = useState(
    song.likes ? song.likes.length > 0 : false,
  );
  const [likesCount, setLikesCount] = useState(song._count.likes);

  const handlePlay = async () => {
    setIsLoading(true);
    const playUrl = await getPlayUrl(song.id);
    setTrack({
      id: song.id,
      title: song.title,
      url: playUrl,
      artwork: song.thumbnailUrl,
      prompt: song.prompt,
      createdByUserName: song.user.name,
    });
    setIsLoading(false);
  };

  const handleLike = async (e: React.MouseEvent) => {
    e.stopPropagation();
    setIsLiked(!isLiked);
    setLikesCount(isLiked ? likesCount - 1 : likesCount + 1);
    await toggleLikeSong(song.id);
  };

  return (
    <div className="group flex flex-col gap-2">
      <div
        onClick={handlePlay}
        className="relative aspect-square w-full cursor-pointer overflow-hidden rounded-lg bg-white/5 shadow-lg transition-transform duration-200 group-hover:scale-[1.03]"
      >
        {song.thumbnailUrl ? (
          <img
            className="h-full w-full object-cover object-center transition-opacity duration-200 group-hover:opacity-80"
            src={song.thumbnailUrl}
            alt={song.title}
          />
        ) : (
          <div className="flex h-full w-full items-center justify-center bg-white/5">
            <Music className="h-12 w-12 text-white/20" />
          </div>
        )}

        {/* Play overlay */}
        <div className="absolute inset-0 flex items-center justify-center bg-black/40 opacity-0 transition-opacity duration-200 group-hover:opacity-100">
          <div className="flex h-12 w-12 items-center justify-center rounded-full bg-white shadow-[0_0_20px_rgba(255,255,255,0.3)] transition-transform duration-150 group-hover:scale-110">
            {isLoading ? (
              <Loader2 className="h-5 w-5 animate-spin text-black" />
            ) : (
              <Play className="h-5 w-5 fill-black text-black" />
            )}
          </div>
        </div>
      </div>

      {/* Info */}
      <div className="flex flex-col gap-0.5 px-0.5">
        <p className="truncate text-sm font-semibold text-white">{song.title}</p>
        <p className="truncate text-xs text-white/50">{song.user.name}</p>
        <div className="mt-1 flex items-center justify-between text-xs text-white/40">
          <span>{song.listenCount} plays</span>
          <button
            onClick={handleLike}
            className="flex cursor-pointer items-center gap-1 transition-colors hover:text-white/80"
          >
            <Heart
              className={`h-3.5 w-3.5 transition-colors ${isLiked ? "fill-red-500 text-red-500" : ""}`}
            />
            {likesCount}
          </button>
        </div>
      </div>
    </div>
  );
}