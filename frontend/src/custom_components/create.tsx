"use client"

import { queueSong } from "@/actions/generation"
import { Button } from "@/components/ui/button"

export default function CreateSong(){
    return(
        <Button onClick={queueSong}>
            Create Song
        </Button>
    )
}