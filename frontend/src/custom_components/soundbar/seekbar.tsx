"use client"

import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { usePlayerStore } from "@/stores/use-player"
import { Music, Pause, Play, Volume2Icon } from "lucide-react";
import { useState } from "react";

export default function Seekbar(){
    const { track } = usePlayerStore();
    const [isPlaying, setIsPlaying] = useState(false);
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
                            </div>
                        </div>

                    </div>

                </div>

            </Card>

        </div>
    )
}