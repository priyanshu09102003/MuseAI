"use client";

import { Button } from "@/components/ui/button";
import { authClient } from "@/lib/auth-client";
import { Crown } from "lucide-react";

export default function Upgrade(){
    const upgrade = async () => {
    await authClient.checkout({
      products: [
        "1d9c9bdd-f713-4b35-b712-92d744191b99",
        "9ae76012-fdcc-4362-981b-69368912054f",
        "1e617678-f389-499b-ae3c-4d94e5a03992",
      ],
    });
  };
    return(
        <Button variant={"outline"} size="sm" className="text-xs font-semibold ml-2 cursor-pointer text-amber-300 hover:bg-amber-300 hover:text-amber-800" onClick={upgrade}>
            <Crown size="4" /> MuseAI Premium
        </Button>
    )
}