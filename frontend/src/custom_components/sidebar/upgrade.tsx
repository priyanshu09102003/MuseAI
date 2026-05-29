"use client";

import { Button } from "@/components/ui/button";
import { Crown } from "lucide-react";

export default function Upgrade(){
    return(
        <Button variant={"outline"} size="sm" className="text-xs font-semibold ml-2 cursor-pointer text-amber-300 hover:bg-amber-300 hover:text-amber-800">
            <Crown size="4" /> MuseAI Premium
        </Button>
    )
}