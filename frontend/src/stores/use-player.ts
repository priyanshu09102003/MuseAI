import { create } from "zustand"

type TrackType = {
  id: string;
  title: string | null;
  url: string | null;
  artwork?: string | null;
  prompt: string | null;
  createdByUserName: string | null;
}

interface PlayerState {
  track: TrackType | null;
  queue: TrackType[];          
  queueIndex: number;          
  setTrack: (track: TrackType) => void;
  setQueue: (tracks: TrackType[], startIndex?: number) => void;
  playNext: () => void;
}

export const usePlayerStore = create<PlayerState>((set, get) => ({
  track: null,
  queue: [],
  queueIndex: 0,

  setTrack: (track) => set({ track }),

  setQueue: (tracks, startIndex = 0) => {
    set({ queue: tracks, queueIndex: startIndex, track: tracks[startIndex] ?? null });
  },

  playNext: () => {
    const { queue, queueIndex } = get();
    const next = queueIndex + 1;
    if (next < queue.length) {
      set({ queueIndex: next, track: queue[next] });
    }
  },
}));