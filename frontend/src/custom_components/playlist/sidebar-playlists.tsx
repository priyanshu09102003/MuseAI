"use client";

import { getUserPlaylists, createPlaylist, deletePlaylist, renamePlaylist } from "@/actions/playlist";
import { SidebarMenuItem, SidebarMenuButton } from "@/components/ui/sidebar";
import { Heart, ListMusic, Plus, MoreHorizontal, Trash2, Pencil, Check, X, Music2 } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { usePlayerStore } from "@/stores/use-player";
import { toast } from "sonner";

type Playlist = { id: string; name: string; _count: { songs: number } };

export function SidebarPlaylists() {
  const [playlists, setPlaylists] = useState<Playlist[]>([]);
  const [creating, setCreating] = useState(false);
  const [newName, setNewName] = useState("");
  const [renamingId, setRenamingId] = useState<string | null>(null);
  const [renameValue, setRenameValue] = useState("");
  const pathname = usePathname();
  const router = useRouter();
  const { track } = usePlayerStore();

  const createRef = useRef<HTMLDivElement>(null);

  const load = () =>
    getUserPlaylists().then((p) => setPlaylists(p as Playlist[]));

  useEffect(() => { load(); }, []);

  // Close create input on click outside
  useEffect(() => {
    if (!creating) return;
    const handler = (e: MouseEvent) => {
      if (createRef.current && !createRef.current.contains(e.target as Node)) {
        setCreating(false);
        setNewName("");
      }
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, [creating]);

  const handleCreate = async () => {
    if (!newName.trim()) return;
    const tempId = `temp-${Date.now()}`;
    const optimistic: Playlist = { id: tempId, name: newName.trim(), _count: { songs: 0 } };

    setPlaylists((prev) => [optimistic, ...prev]);
    setNewName("");
    setCreating(false);
    toast.success(`Playlist "${optimistic.name}" created`);

    const real = await createPlaylist(optimistic.name);
    setPlaylists((prev) =>
      prev.map((p) => (p.id === tempId ? { ...p, id: real.id } : p))
    );
  };

  const handleDelete = async (playlistId: string) => {
    setPlaylists((prev) => prev.filter((p) => p.id !== playlistId));

    if (pathname === `/playlist/${playlistId}`) {
      router.push("/");
    }

    await deletePlaylist(playlistId);
  };

  const handleRenameStart = (pl: Playlist) => {
    setRenamingId(pl.id);
    setRenameValue(pl.name);
  };

  const handleRenameSubmit = async (playlistId: string) => {
    if (!renameValue.trim()) return;
    const name = renameValue.trim();

    setPlaylists((prev) =>
      prev.map((p) => (p.id === playlistId ? { ...p, name } : p))
    );
    setRenamingId(null);

    await renamePlaylist(playlistId, name);
  };

  const likedSongs = playlists.find((p) => p.name === "Liked Songs");
  const userPlaylists = playlists.filter((p) => p.name !== "Liked Songs");

  const isPlayingInPlaylist = (playlistId: string) =>
    pathname === `/playlist/${playlistId}` && !!track;

  return (
    <div className="mt-2 mx-2 flex flex-col min-h-0 rounded-xl bg-white/[0.02] border border-white/5 px-1 pt-1 pb-2">
      {/* Header */}
      <div className="flex items-center justify-between px-2 py-1 shrink-0">
        <span className="text-xs font-semibold uppercase tracking-wider text-white/60">
          Playlists
        </span>
        <button
          onClick={() => setCreating(true)}
          className="rounded-md p-1 hover:bg-white/5 transition-colors cursor-pointer"
        >
          <Plus className="h-3.5 w-3.5 text-white/40" />
        </button>
      </div>

      {/* Create input */}
      {creating && (
        <div ref={createRef} className="flex gap-1.5 px-1 py-1.5 shrink-0">
          <Input
            autoFocus
            size={1}
            placeholder="Playlist name"
            value={newName}
            onChange={(e) => setNewName(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") handleCreate();
              if (e.key === "Escape") setCreating(false);
            }}
            className="h-7 text-xs flex-1"
          />
          <Button size="sm" className="h-7 px-2 text-xs cursor-pointer" onClick={handleCreate}>
            Add
          </Button>
        </div>
      )}

      {/* Scrollable list */}
      <ul className="mt-1 space-y-1 overflow-y-auto flex-1 max-h-[40vh] pr-0.5
        scrollbar-thin scrollbar-thumb-white/10 scrollbar-track-transparent">

        {/* Liked Songs — pinned, no rename/delete */}
        {likedSongs && (
          <SidebarMenuItem>
            <SidebarMenuButton
              isActive={pathname === `/playlist/${likedSongs.id}`}
              className={`group flex items-center gap-2.5 rounded-md px-2 py-2 text-sm transition-colors w-full
                ${pathname === `/playlist/${likedSongs.id}` ? "bg-primary/20 text-primary" : "text-white/60 hover:bg-white/5 hover:text-white"}`}
            >
              <Link href={`/playlist/${likedSongs.id}`} className="flex items-center gap-2.5 w-full min-w-0">
                <div className="flex h-7 w-7 shrink-0 items-center justify-center rounded-md bg-gradient-to-br from-indigo-500 to-purple-500">
                  <Heart className="h-3.5 w-3.5 fill-white text-white" />
                </div>
                <span className="truncate text-xs flex-1">Liked Songs</span>
                {isPlayingInPlaylist(likedSongs.id) && (
                  <Music2 className="h-3 w-3 shrink-0 text-green-400 animate-pulse" />
                )}
              </Link>
            </SidebarMenuButton>
          </SidebarMenuItem>
        )}

        {/* User playlists */}
        {userPlaylists.map((pl) => (
          <SidebarMenuItem key={pl.id}>
            <div className={`group flex items-center gap-1 rounded-md px-2 py-2 text-sm transition-colors w-full
              ${pathname === `/playlist/${pl.id}` ? "bg-primary/20 text-primary" : "text-white/60 hover:bg-white/5 hover:text-white"}`}>

              {renamingId === pl.id ? (
                <div className="flex items-center gap-1 w-full">
                  <Input
                    autoFocus
                    size={1}
                    value={renameValue}
                    onChange={(e) => setRenameValue(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === "Enter") handleRenameSubmit(pl.id);
                      if (e.key === "Escape") setRenamingId(null);
                    }}
                    className="h-6 text-xs flex-1 px-1"
                  />
                  <button onClick={() => handleRenameSubmit(pl.id)} className="p-0.5 hover:text-green-400">
                    <Check className="h-3.5 w-3.5" />
                  </button>
                  <button onClick={() => setRenamingId(null)} className="p-0.5 hover:text-red-400">
                    <X className="h-3.5 w-3.5" />
                  </button>
                </div>
              ) : (
                <>
                  <Link href={`/playlist/${pl.id}`} className="flex items-center gap-2.5 flex-1 min-w-0">
                    <div className="flex h-7 w-7 shrink-0 items-center justify-center rounded-md bg-white/10">
                      <ListMusic className="h-3.5 w-3.5" />
                    </div>
                    <span className="truncate text-xs flex-1">{pl.name}</span>
                    {isPlayingInPlaylist(pl.id) && (
                      <Music2 className="h-3 w-3 shrink-0 text-green-400 animate-pulse" />
                    )}
                  </Link>

                  <DropdownMenu>
                    <DropdownMenuTrigger>
                      <button className="opacity-0 group-hover:opacity-100 transition-opacity shrink-0 p-0.5 rounded hover:bg-white/10 cursor-pointer">
                        <MoreHorizontal className="h-3.5 w-3.5" />
                      </button>
                    </DropdownMenuTrigger>
                    <DropdownMenuContent align="end" className="w-36">
                      <DropdownMenuItem onClick={() => handleRenameStart(pl)}>
                        <Pencil className="mr-2 h-3.5 w-3.5" />
                        Rename
                      </DropdownMenuItem>
                      <DropdownMenuItem
                        onClick={() => handleDelete(pl.id)}
                        className="text-red-400 focus:text-red-400"
                      >
                        <Trash2 className="mr-2 h-3.5 w-3.5" />
                        Delete
                      </DropdownMenuItem>
                    </DropdownMenuContent>
                  </DropdownMenu>
                </>
              )}
            </div>
          </SidebarMenuItem>
        ))}
      </ul>
    </div>
  );
}