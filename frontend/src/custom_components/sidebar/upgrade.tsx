"use client";

import { Button } from "@/components/ui/button";
import { authClient } from "@/lib/auth-client";
import { Crown } from "lucide-react";

export default function Upgrade(){
    const upgrade = async () => {
    await authClient.checkout({
      products: [
        "afdfc695-519e-47a1-a469-918d28e6c4c8",
        "13a83be1-0727-4f28-aec4-bd5d935d8ed0",
        "cec41d08-7a3b-41df-bc2d-f756cc7eacc3",
      ],
    });
  };
    return(
        <Button variant={"outline"} size="sm" className="text-xs font-semibold ml-2 cursor-pointer text-amber-300 hover:bg-amber-300 hover:text-amber-800" onClick={upgrade}>
            <Crown size="4" /> MuseAI Premium
        </Button>
    )
}