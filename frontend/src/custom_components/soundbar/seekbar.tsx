"use client"

import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from "@/components/ui/dropdown-menu";
import { Slider } from "@/components/ui/slider";
import { usePlayerStore } from "@/stores/use-player"
import { Download, MoreHorizontal, Music, Pause, Play, Volume2Icon, VolumeX } from "lucide-react";
import { useEffect, useRef, useState } from "react";

export default function Seekbar(){
    const { track } = usePlayerStore();
    const [isPlaying, setIsPlaying] = useState(false);
    const [currentTime, setCurrentTime] = useState(0);
    const [volume, setVolume] = useState(100)
    const [duration, setDuration] = useState(0);
    const [isMuted, setIsMuted] = useState(false);
    const [prevVolume, setPrevVolume] = useState(100);

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


    const togglePlay = ()  => {
        if(!track?.url || !audioRef.current)return;

        if(isPlaying){
            audioRef.current.pause()
            setIsPlaying(false)
        }

        else{
            audioRef.current.play()
            setIsPlaying(true)
        }
    }

    const toggleMute = () => {
    if (!audioRef.current) return;
    if (isMuted) {
        audioRef.current.volume = prevVolume / 100;
        setVolume(prevVolume);
        setIsMuted(false);
    } else {
        setPrevVolume(volume);
        audioRef.current.volume = 0;
        setVolume(0);
        setIsMuted(true);
    }
};

    const handleSeek = (value: number | readonly number[]) => {
        const val = Array.isArray(value) ? value[0] : value;
        if (audioRef.current && val !== undefined) {
            audioRef.current.currentTime = val;
            setCurrentTime(val);
        }
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

                                <Button variant="ghost" className="cursor-pointer" size="icon" onClick={togglePlay}>

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

                                <button onClick={toggleMute} className="cursor-pointer">
                                    {isMuted || volume === 0 
                                        ? <VolumeX className="h-4 w-4" /> 
                                        : <Volume2Icon className="h-4 w-4" />
                                    }
                                </button>
                               <input
                                type="range"
                                min={0}
                                max={100}
                                step={1}
                                value={volume}
                                onChange={(e) => {
                                    const v = Number(e.target.value);
                                    setVolume(v);
                                    if (audioRef.current) audioRef.current.volume = v / 100;
                                }}
                                className="w-20 accent-white cursor-pointer"
                            />
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

                        <span className="text-muted-foreground w-8 text-left text-[10px]">
                            {formatTime(duration)}
                        </span>

                    </div>

                </div>


                {track?.url && (
                    <audio
                        ref={audioRef}
                        preload="metadata"
                        onTimeUpdate={() => setCurrentTime(audioRef.current?.currentTime ?? 0)}
                        onLoadedMetadata={() => setDuration(audioRef.current?.duration ?? 0)}
                        onEnded={() => {
                        const { queue, queueIndex, playNext } = usePlayerStore.getState();
                        if (queueIndex < queue.length - 1) {
                            playNext();
                        } else {
                            setIsPlaying(false);
                            setCurrentTime(0);
                        }
                        }}
                    />
                )}

            </Card>

        </div>
    )
}