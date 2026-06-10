"use client";

import { useState, useEffect } from "react";
import {
  Dialog, DialogContent, DialogHeader, DialogTitle,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { getUserPlaylists, createPlaylist, addSongToPlaylist } from "@/actions/playlist";
import { ListMusic, Plus, Check } from "lucide-react";
import { toast } from "sonner";

type Playlist = { id: string; name: string; _count: { songs: number } };

export function AddToPlaylistDialog({
  songId,
  open,
  onClose,
}: {
  songId: string;
  open: boolean;
  onClose: () => void;
}) {
  const [playlists, setPlaylists] = useState<Playlist[]>([]);
  const [newName, setNewName] = useState("");
  const [creating, setCreating] = useState(false);
  const [loading, setLoading] = useState(false);
  const [added, setAdded] = useState<string | null>(null);

  useEffect(() => {
    if (open) {
      getUserPlaylists().then((p) =>
        setPlaylists(p.filter((pl) => pl.name !== "Liked Songs"))
      );
    }
  }, [open]);

  const handleAdd = (playlistId: string, playlistName: string) => {
    setAdded(playlistId);
    setTimeout(onClose, 400);
    addSongToPlaylist(playlistId, songId);
    toast.success(`Added to ${playlistName}`);
  };

  const handleCreate = () => {
    if (!newName.trim()) return;
    const name = newName.trim();
    onClose();
    createPlaylist(name).then((pl) => addSongToPlaylist(pl.id, songId));
    toast.success(`Playlist "${name}" created`);
  };

  return (
    <Dialog open={open} onOpenChange={onClose}>
      <DialogContent className="max-w-sm">
        <DialogHeader>
          <DialogTitle>Add to playlist</DialogTitle>
        </DialogHeader>

        <div className="space-y-2 mt-2">
          {playlists.length === 0 && !creating && (
            <p className="text-muted-foreground text-sm text-center py-4">
              No playlists yet. Create one below.
            </p>
          )}

          {playlists.map((pl) => (
            <button
              key={pl.id}
              onClick={() => handleAdd(pl.id, pl.name)}
              disabled={loading}
              className="w-full flex items-center gap-3 rounded-lg px-3 py-2.5 text-sm hover:bg-white/5 transition-colors text-left"
            >
              <div className="flex h-9 w-9 items-center justify-center rounded-md bg-white/10 shrink-0">
                <ListMusic className="h-4 w-4" />
              </div>
              <div className="flex-1 min-w-0">
                <p className="font-medium truncate">{pl.name}</p>
                <p className="text-xs text-muted-foreground">{pl._count.songs} songs</p>
              </div>
              {added === pl.id && <Check className="h-4 w-4 text-green-500 shrink-0" />}
            </button>
          ))}

          {creating ? (
            <div className="flex gap-2 pt-1">
              <Input
                autoFocus
                placeholder="Playlist name"
                value={newName}
                onChange={(e) => setNewName(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && handleCreate()}
                className="flex-1"
              />
              <Button size="sm" onClick={handleCreate} disabled={!newName.trim()}>
                Create
              </Button>
              <Button size="sm" variant="ghost" onClick={() => setCreating(false)}>
                Cancel
              </Button>
            </div>
          ) : (
            <button
              onClick={() => setCreating(true)}
              className="w-full flex items-center gap-3 rounded-lg px-3 py-2.5 text-sm hover:bg-white/5 transition-colors text-muted-foreground"
            >
              <div className="flex h-9 w-9 items-center justify-center rounded-md border border-dashed border-white/20 shrink-0">
                <Plus className="h-4 w-4" />
              </div>
              New playlist
            </button>
          )}
        </div>
      </DialogContent>
    </Dialog>
  );
}