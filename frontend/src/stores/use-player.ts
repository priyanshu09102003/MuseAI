interface PlayerTrack {
  id: string;
  title: string | null;
  url: string | null;
  artwork?: string | null;
  prompt: string | null;
  createdByUserName: string | null;
}

interface PlayerState{
    track: PlayerTrack | null;
    setTrack: (track: PlayerTrack) => void;
}