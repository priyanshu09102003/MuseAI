"use client"

import { getPlayUrl } from "@/actions/generation";
import { renameSong, setPublishedStatus } from "@/actions/song";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"
import { Input } from "@/components/ui/input";
import {
  DownloadIcon,
  Loader2,
  MoreHorizontalIcon,
  Music,
  PencilIcon,
  Play,
  RefreshCcw,
  Search,
  XCircle,
} from "lucide-react";
import { useState } from "react";
import { RenameDialog } from "./rename-dialog";
import { useRouter } from "next/navigation";

export interface Track {
  id: string;
  title: string | null;
  createdAt: Date;
  instrumental: boolean;
  prompt: string | null;
  lyrics: string | null;
  describedLyrics: string | null;
  fullDescribedSong: string | null;
  thumbnailUrl: string | null;
  playUrl: string | null;
  status: string | null;
  createdByUserName: string | null;
  published: boolean;
}

export function TracksList({ tracks }: { tracks: Track[] }) {
  const [searchQuery, setSearchQuery] = useState("");
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [loadingTrackId, setLoadingTrackId] = useState<string | null>(null);
  const [trackToRename, setTrackToRename] = useState<Track | null>(null);
  const router = useRouter();

  
  const [localTracks, setLocalTracks] = useState<Track[]>(tracks);

  const handleRefresh = async () => {
    setIsRefreshing(true);
    router.refresh();
    setTimeout(() => setIsRefreshing(false), 1000);
  };

  const handleTrackSelect = async (track: Track) => {
    if (loadingTrackId) return;
    setLoadingTrackId(track.id);
    const playUrl = await getPlayUrl(track.id);
    setLoadingTrackId(null);
  };

  
  const handleRename = async (trackId: string, newTitle: string) => {
    setLocalTracks((prev) =>
      prev.map((t) => (t.id === trackId ? { ...t, title: newTitle } : t))
    );
    await renameSong(trackId, newTitle);
  };

  const filteredTracks = localTracks.filter(
    (track) =>
      track.title?.toLowerCase().includes(searchQuery.toLowerCase()) ||
      track.prompt?.toLowerCase().includes(searchQuery.toLowerCase())
  );

  return (
    <div className="flex flex-1 flex-col overflow-y-scroll">
      <div className="flex-1 p-6">
        {/* Search + Refresh */}
        <div className="mb-4 flex items-center justify-between gap-4">
          <div className="relative max-w-md flex-1">
            <Search className="text-muted-foreground absolute top-1/2 left-3 h-4 w-4 -translate-y-1/2" />
            <Input
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Search..."
              className="pl-10"
            />
          </div>

          <Button
            disabled={isRefreshing}
            variant="outline"
            size="sm"
            className="cursor-pointer font-semibold"
            onClick={handleRefresh}
          >
            {isRefreshing ? (
              <Loader2 className="mr-2 animate-spin" />
            ) : (
              <RefreshCcw className="mr-2" />
            )}
            Refresh
          </Button>
        </div>

        {/* Track list */}
        <div className="space-y-2">
          {filteredTracks.length > 0 ? (
            filteredTracks.map((track) => {
              switch (track.status) {
                case "failed":
                  return (
                    <div
                      key={track.id}
                      className="flex cursor-not-allowed items-center gap-4 rounded-lg p-3"
                    >
                      <div className="bg-destructive/10 flex h-12 w-12 shrink-0 items-center justify-center rounded-md">
                        <XCircle className="text-destructive h-6 w-6" />
                      </div>
                      <div className="min-w-0 flex-1">
                        <h3 className="text-destructive truncate text-sm font-medium">
                          Generation failed
                        </h3>
                        <p className="text-muted-foreground truncate text-xs">
                          Please try again later.
                        </p>
                      </div>
                    </div>
                  );

                case "no credits":
                  return (
                    <div
                      key={track.id}
                      className="flex cursor-not-allowed items-center gap-4 rounded-lg p-3"
                    >
                      <div className="bg-destructive/10 flex h-12 w-12 shrink-0 items-center justify-center rounded-md">
                        <XCircle className="text-destructive h-6 w-6" />
                      </div>
                      <div className="min-w-0 flex-1">
                        <h3 className="text-destructive truncate text-sm font-medium">
                          Not enough credits
                        </h3>
                        <p className="text-muted-foreground truncate text-xs">
                          Please upgrade to MuseAI Premium to gain more credits.
                        </p>
                      </div>
                    </div>
                  );

                case "queued":
                case "processing":
                  return (
                    <div
                      key={track.id}
                      className="flex cursor-not-allowed items-center gap-4 rounded-lg p-3"
                    >
                      <div className="bg-muted flex h-12 w-12 shrink-0 items-center justify-center rounded-md">
                        <Loader2 className="text-muted-foreground h-6 w-6 animate-spin" />
                      </div>
                      <div className="min-w-0 flex-1">
                        <h3 className="text-muted-foreground truncate text-sm font-medium">
                          Processing song...
                        </h3>
                        <p className="text-muted-foreground truncate text-xs">
                          Refresh to check the generation status.
                        </p>
                      </div>
                    </div>
                  );

                default:
                  return (
                    <div
                      key={track.id}
                      className="hover:bg-muted/50 flex cursor-pointer items-center gap-4 rounded-lg p-3 transition-colors"
                      onClick={() => handleTrackSelect(track)}
                    >
                      {/* Thumbnail */}
                      <div className="group relative h-12 w-12 shrink-0 overflow-hidden rounded-md">
                        {track.thumbnailUrl ? (
                          <img
                            src={track.thumbnailUrl}
                            alt="thumbnail"
                            className="h-full w-full object-cover"
                          />
                        ) : (
                          <div className="bg-muted flex h-full w-full items-center justify-center">
                            <Music className="text-muted-foreground h-6 w-6" />
                          </div>
                        )}
                        <div className="absolute inset-0 flex items-center justify-center bg-black/20 opacity-0 transition-opacity group-hover:opacity-100">
                          {loadingTrackId === track.id ? (
                            <Loader2 className="animate-spin text-white" />
                          ) : (
                            <Play className="fill-white text-white" />
                          )}
                        </div>
                      </div>

                      {/* Track info */}
                      <div className="min-w-0 flex-1">
                        <div className="flex items-center gap-2">
                          <h3 className="truncate text-sm font-medium">
                            {track.title}
                          </h3>
                          {track.instrumental && (
                            <Badge variant="secondary">Instrumental</Badge>
                          )}
                        </div>
                        <p className="text-muted-foreground truncate text-xs">
                          {track.prompt}
                        </p>
                      </div>

                      {/* Actions */}
                      <div className="flex items-center gap-2">
                        <Button
                          variant="secondary"
                          size="sm"
                          className="cursor-pointer"
                          onClick={async (e) => {
                            e.stopPropagation();
                            await setPublishedStatus(track.id, !track.published);
                          }}
                        >
                          {track.published ? "Unpublish" : "Publish"}
                        </Button>

                        <DropdownMenu>
                          <DropdownMenuTrigger>
                            <Button
                              variant="outline"
                              size="icon"
                              className="cursor-pointer hover:bg-white hover:text-black transition-all"
                              onClick={(e) => e.stopPropagation()}
                            >
                              <MoreHorizontalIcon />
                            </Button>
                          </DropdownMenuTrigger>

                          <DropdownMenuContent align="end" className="w-45">
                            <DropdownMenuItem
                              className="cursor-pointer"
                              onClick={async (e) => {
                                e.stopPropagation();
                                const playUrl = await getPlayUrl(track.id);
                                window.open(playUrl, "_blank");
                              }}
                            >
                              <DownloadIcon className="mr-2" /> Download Track
                            </DropdownMenuItem>

                            <DropdownMenuItem
                              className="cursor-pointer"
                              onClick={(e) => {
                                e.stopPropagation();
                                setTrackToRename(track);
                              }}
                            >
                              <PencilIcon className="mr-2" /> Rename Track
                            </DropdownMenuItem>
                          </DropdownMenuContent>
                        </DropdownMenu>
                      </div>
                    </div>
                  );
              }
            })):(
              <div className="flex flex-col items-center justify-center pt-20 text-center">
                  <Music className="text-muted-foreground h-10 w-10" />
                  <h2 className="mt-4 text-lg font-semibold">No creations yet</h2>
                  <p className="text-muted-foreground mt-1 text-sm">
                    {searchQuery
                    ? "No tracks match your search."
                    : "Create your first song to get started."}
                  </p>
              </div>
            )}
        </div>
      </div>

      {/* Rename dialog */}
      {trackToRename && (
        <RenameDialog
          track={trackToRename}
          onClose={() => setTrackToRename(null)}
          onRename={handleRename}
        />
      )}
    </div>
  );
}