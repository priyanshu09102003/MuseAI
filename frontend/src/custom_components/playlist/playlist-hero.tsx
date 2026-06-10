"use client";

import { getPlayUrl } from "@/actions/generation";
import { removeSongFromPlaylist } from "@/actions/playlist";
import { usePlayerStore } from "@/stores/use-player";
import { Music, Play, Clock, Heart, Loader2, Trash2, Music2 } from "lucide-react";
import { useState } from "react";
import { toggleLikeSong } from "@/actions/song";

type Song = {
  id: string; title: string; thumbnailUrl?: string | null;
  prompt: string | null; audioDuration: number | null;
  user: { name: string | null }; _count: { likes: number };
  likes?: { userId: string }[];
};

type Playlist = { id: string; name: string; createdAt: Date };

export function PlaylistHero({
  playlist, songs: initialSongs, songCount, durationStr,
}: {
  playlist: Playlist;
  songs: Song[];
  songCount: number;
  durationStr: string;
}) {
  const { setQueue, setTrack, track } = usePlayerStore();
  const [songs, setSongs] = useState<Song[]>(initialSongs);
  const [loadingId, setLoadingId] = useState<string | null>(null);
  // Optimistic like state
  const [likedIds, setLikedIds] = useState<Set<string>>(
    new Set(initialSongs.filter((s) => s.likes && s.likes.length > 0).map((s) => s.id))
  );

  const thumbs = songs.slice(0, 4).map((s) => s.thumbnailUrl).filter(Boolean);

  
  const handlePlayAll = async () => {
    if (songs.length === 0) return;
    setLoadingId("all");
    const tracks = await Promise.all(
      songs.map(async (s) => {
        const url = await getPlayUrl(s.id);
        return {
          id: s.id, title: s.title, url,
          artwork: s.thumbnailUrl, prompt: s.prompt,
          createdByUserName: s.user.name,
        };
      })
    );
    setQueue(tracks, 0);
    setLoadingId(null);
  };

 
  const handlePlaySong = async (song: Song, index: number) => {
    setLoadingId(song.id);
    const url = await getPlayUrl(song.id);
    setTrack({
      id: song.id, title: song.title, url,
      artwork: song.thumbnailUrl, prompt: song.prompt,
      createdByUserName: song.user.name,
    });
    setLoadingId(null);
  };

  
  const handleRemove = async (e: React.MouseEvent, songId: string) => {
    e.stopPropagation();
    // 1. Remove from UI immediately
    setSongs((prev) => prev.filter((s) => s.id !== songId));
    // 2. Persist in background — no await needed for UI
    removeSongFromPlaylist(playlist.id, songId);
  };

  
  const handleLike = async (e: React.MouseEvent, songId: string) => {
    e.stopPropagation();
    // 1. Toggle immediately
    setLikedIds((prev) => {
      const next = new Set(prev);
      next.has(songId) ? next.delete(songId) : next.add(songId);
      return next;
    });
    // 2. Persist in background
    toggleLikeSong(songId);
  };

  const formatDur = (sec: number | null) => {
    if (!sec) return "--:--";
    const m = Math.floor(sec / 60), s = Math.floor(sec % 60);
    return `${m}:${s.toString().padStart(2, "0")}`;
  };

  return (
    <div>
      {/* Hero */}
      <div className="flex flex-col gap-6 bg-gradient-to-b from-purple-900/60 to-transparent p-6 sm:flex-row sm:items-end">
        {/* Mosaic thumbnail */}
        <div className="h-44 w-44 shrink-0 overflow-hidden rounded-xl shadow-2xl sm:h-52 sm:w-52">
          {thumbs.length === 0 ? (
            <div className="flex h-full w-full items-center justify-center bg-white/10">
              <Music className="h-16 w-16 text-white/20" />
            </div>
          ) : thumbs.length < 4 ? (
            <img src={thumbs[0]!} className="h-full w-full object-cover" alt="" />
          ) : (
            <div className="grid h-full w-full grid-cols-2 grid-rows-2">
              {thumbs.map((t, i) => (
                <img key={i} src={t!} className="h-full w-full object-cover" alt="" />
              ))}
            </div>
          )}
        </div>

        {/* Info — song count reads from live local state */}
        <div className="flex flex-col gap-2">
          <p className="text-xs font-semibold uppercase tracking-widest text-white/50">Playlist</p>
          <h1 className="text-3xl font-bold text-white sm:text-5xl">{playlist.name}</h1>
          <p className="text-sm text-white/50">
            {songs.length} songs · {durationStr}
          </p>
        </div>
      </div>

      {/* Play button */}
      <div className="flex items-center gap-4 px-6 py-4">
        <button
          onClick={handlePlayAll}
          disabled={loadingId === "all" || songs.length === 0}
          className="flex h-14 w-14 items-center justify-center rounded-full bg-green-500 shadow-lg transition-transform hover:scale-105 active:scale-95 disabled:opacity-60"
        >
          {loadingId === "all"
            ? <Loader2 className="h-6 w-6 animate-spin text-black" />
            : <Play className="h-6 w-6 fill-black text-black ml-0.5" />
          }
        </button>
      </div>

      {/* Track list */}
      {songs.length === 0 ? (
        <div className="flex flex-col items-center justify-center py-20 text-center">
          <Music className="h-12 w-12 text-white/20" />
          <p className="mt-4 text-white/40">No songs in this playlist yet.</p>
        </div>
      ) : (
        <div className="px-6">
          {/* Header */}
          <div className="mb-2 grid grid-cols-[2rem_1fr_1fr_6rem] gap-4 border-b border-white/10 pb-2 text-xs font-medium uppercase tracking-wider text-white/40">
            <span className="text-center">#</span>
            <span>Title</span>
            <span className="hidden sm:block">Artist</span>
            <span className="flex justify-end"><Clock className="h-3.5 w-3.5" /></span>
          </div>

          <div className="space-y-0.5">
            {songs.map((song, i) => {
              const isCurrentlyPlaying = track?.id === song.id;

              return (
                <div
                  key={song.id}
                  onClick={() => handlePlaySong(song, i)}
                  className={`group grid cursor-pointer grid-cols-[2rem_1fr_1fr_6rem] gap-4 rounded-md px-0 py-2 transition-colors items-center
                    ${isCurrentlyPlaying ? "bg-white/5" : "hover:bg-white/5"}`}
                >
                  {/* Index / spinner / now-playing */}
                  <div className="flex items-center justify-center">
                    {isCurrentlyPlaying ? (
                      <Music2 className="h-4 w-4 text-green-400 animate-pulse" />
                    ) : (
                      <>
                        <span className="text-sm text-white/40 group-hover:hidden">{i + 1}</span>
                        <span className="hidden group-hover:flex items-center justify-center">
                          {loadingId === song.id
                            ? <Loader2 className="h-4 w-4 animate-spin text-white" />
                            : <Play className="h-4 w-4 fill-white text-white" />
                          }
                        </span>
                      </>
                    )}
                  </div>

                  {/* Title + thumbnail */}
                  <div className="flex items-center gap-3 min-w-0">
                    <div className="h-10 w-10 shrink-0 overflow-hidden rounded-md bg-white/5">
                      {song.thumbnailUrl
                        ? <img src={song.thumbnailUrl} className="h-full w-full object-cover" alt="" />
                        : <div className="flex h-full w-full items-center justify-center"><Music className="h-4 w-4 text-white/20" /></div>
                      }
                    </div>
                    <p className={`truncate text-sm font-medium ${isCurrentlyPlaying ? "text-green-400" : "text-white"}`}>
                      {song.title}
                    </p>
                  </div>

                  {/* Artist */}
                  <p className="hidden truncate text-sm text-white/50 sm:block">{song.user.name}</p>

                  {/* Actions: like + duration + remove */}
                  <div className="flex items-center justify-end gap-2">
                    <button
                      onClick={(e) => handleLike(e, song.id)}
                      className="opacity-0 group-hover:opacity-100 transition-opacity shrink-0"
                    >
                      <Heart className={`h-4 w-4 ${likedIds.has(song.id) ? "fill-red-500 text-red-500" : "text-white/40 hover:text-white"}`} />
                    </button>

                    <span className="text-sm text-white/40 shrink-0">{formatDur(song.audioDuration)}</span>

                    <button
                      onClick={(e) => handleRemove(e, song.id)}
                      className="opacity-0 group-hover:opacity-100 transition-opacity shrink-0"
                      title="Remove from playlist"
                    >
                      <Trash2 className="h-4 w-4 text-white/40 hover:text-red-400 transition-colors" />
                    </button>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}