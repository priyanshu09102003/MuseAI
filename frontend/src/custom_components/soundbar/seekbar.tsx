"use client"

import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from "@/components/ui/dropdown-menu";
import { Slider } from "@/components/ui/slider";
import { usePlayerStore } from "@/stores/use-player"
import { Download, MoreHorizontal, Music, Pause, Play, Volume2Icon } from "lucide-react";
import { useEffect, useRef, useState } from "react";

export default function Seekbar(){
    const { track } = usePlayerStore();
    const [isPlaying, setIsPlaying] = useState(false);
    const [currentTime, setCurrentTime] = useState(0);
    const [volume, setVolume] = useState<number[]>([100])
    const [duration, setDuration] = useState(0);

    const audioRef = useRef<HTMLAudioElement>(null);

    useEffect(() => {
    if (audioRef.current && track?.url) {
            setCurrentTime(0);
            setDuration(0);

            audioRef.current.src = track.url;
            audioRef.current.load();

            const playPromise = audioRef.current.play();
            if (playPromise !== undefined) {
                playPromise
                .then(() => {
                    setIsPlaying(true);
                })
                .catch((error) => {
                    console.error("Playback failed: ", error);
                    setIsPlaying(false);
                });
            }
        }
    }, [track]);


    const handleSeek = (value: number | readonly number[]) => {

    };

    

    const formatTime = (time: number) => {
        const minutes = Math.floor(time / 60);
        const seconds = Math.floor(time % 60);
        return `${minutes.toString().padStart(2, "0")}:${seconds.toString().padStart(2, "0")}`;
    };


    if (!track) return null; 
    return(

        <div className="px-4 pb-2">
            <Card className="bg-white/5 relative w-full shrink-0 border border-white/10 py-0 backdrop-blur-xl shadow-[0_-4px_30px_rgba(139,92,246,0.15)]">
                <div className="space-y-2 p-3">

                    <div className="flex items-center justify-between">

                        <div className="flex min-w-0 flex-1 items-center gap-2">

                            <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-md bg-gradient-to-br from-purple-500 to-pink-500">
                                {
                                    track?.artwork ? <img className="h-full w-full rounded-md object-cover" src={track.artwork} alt="thumbnail" /> : <Music className="text-white" />
                                }
                            </div>

                            <div className="max-w-24 min-w-0 flex-1 md:max-w-full">
                                <p className="truncate text-sm font-semibold">{track?.title || "Untitled"}</p>
                                <p className="text-muted-foreground truncate text-xs">
                                    {track?.createdByUserName || "MuseAI User"}
                                </p>

                            </div>
                        </div>

                        {/* CENTERED CONTROLS */}

                        <div className="absolute left-1/2 -translate-x-1/2">

                                <Button variant="ghost" className="cursor-pointer" size="icon">

                                     {isPlaying ? (
                                        <Pause className="h-4 w-4" />
                                        ) : (
                                        <Play className="h-4 w-4" />
                                        )}
                                </Button>
                        
                        </div>

                        {/* Additional controls */}

                        <div className="flex items-center gap-1">
                            <div className="flex items-center gap-2">

                                <Volume2Icon className="h-4 w-4" />
                                <Slider value={volume} onValueChange={(val) => setVolume(val as number[])}  step={1} min={0} max={100} className="w-24" />
                            </div>

                            <DropdownMenu>
                                <DropdownMenuTrigger>

                                    <Button variant="ghost" size="icon" className="cursor-pointer">
                                        <MoreHorizontal className="h-4 w-4"/> 
                                    </Button>
                                    
                                </DropdownMenuTrigger>

                                <DropdownMenuContent align="end" className="w-40">
                                    <DropdownMenuItem
                                        onClick={() => {
                                        if (!track?.url) return;

                                        window.open(track?.url, "_blank");
                                        }}
                                    >
                                        <Download className="mr-2 h-4 w-4" />
                                        Download
                                    </DropdownMenuItem>
                                </DropdownMenuContent>
                            </DropdownMenu>
                        </div>

                    </div>

                    {/* Full progress bar for the song */}

                    <div className="flex items-center gap-1">
                        <span className="text-muted-foreground w-8 text-right text-[10px]">
                            {formatTime(currentTime)}
                        </span>

                        <Slider className="flex-1" value={[currentTime]} max={duration || 100} step={1} onValueChange={handleSeek}/>

                    </div>

                </div>


                <audio ref={audioRef } src={track.url ?? ""} preload="metadata" />

            </Card>

        </div>
    )
}