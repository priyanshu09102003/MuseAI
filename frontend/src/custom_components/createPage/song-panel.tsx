"use client";

import { GenerateRequest, generateSong } from "@/actions/generation";
import { Button } from "@/components/ui/button";
import { Switch } from "@/components/ui/switch";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Textarea } from "@/components/ui/textarea";
import { Hash, Loader2Icon, Music2Icon, Plus, X } from "lucide-react";
import { useState } from "react";
import { useRouter } from "next/navigation";
import { toast } from "sonner";

const inspirationTags = [
  "80s synth-pop",
  "Acoustic ballad",
  "Epic movie score",
  "Lo-fi hip hop",
  "Driving rock anthem",
  "Summer beach vibe",
  "Dark jazz noir",
  "Bollywood fusion",
  "Cyberpunk electronic",
  "Peaceful piano solo",
  "Trap soul R&B",
  "Celtic folk adventure",
  "Midnight city pop",
  "Orchestral battle theme",
  "Reggaeton banger",
  "Dreamy shoegaze",
  "Afrobeats groove",
  "Anime opening",
  "Chill study beats",
  "Country road trip",
];

const styleTags = [
  "Industrial rave",
  "Heavy bass",
  "Orchestral",
  "Electronic beats",
  "Funky guitar",
  "Soulful vocals",
  "Ambient pads",
  "Distorted synth",
  "Jazzy chords",
  "Punchy drums",
  "Warm bassline",
  "Ethereal vocals",
  "Glitchy FX",
  "Vintage keys",
  "Sub bass",
  "Cinematic strings",
  "Hypnotic groove",
  "Raw acoustic",
];

export function SongPanel(){
    const [mode, setMode] = useState<"simple"|"custom">("simple");
    const [description, setDescription] = useState("")
    const [instrumental, setInstrumental] = useState(false);
    const [lyricsMode, setLyricsMode] = useState<"write"|"auto">("write");
    const [lyrics, setLyrics] = useState("")
    const [styleInput, setStyleInput] = useState("")
    const [loading, setLoading] = useState(false);
    const router = useRouter();

    const handleInspirationTagClick = (tag: string) => {
        const currentTags = description
            .split(", ")
            .map((s) => s.trim())
            .filter((s) => s);

        if (currentTags.includes(tag)) {
            // Deselect — remove from description
            const newTags = currentTags.filter((t) => t !== tag);
            setDescription(newTags.join(", "));
        } else {
            // Select — add to description
            if (description.trim() === "") {
                setDescription(tag);
            } else {
                setDescription(description + ", " + tag);
            }
        }
    };

    const handleStyleInputTagClick = (tag: string) => {
        const currentTags = styleInput
            .split(", ")
            .map((s) => s.trim())
            .filter((s) => s);

        if (currentTags.includes(tag)) {
            const newTags = currentTags.filter((t) => t !== tag);
            setStyleInput(newTags.join(", "));
        } else {
            if (styleInput.trim() === "") {
                setStyleInput(tag);
            } else {
                setStyleInput(styleInput + ", " + tag);
            }
        }
    };

    const handleCreate = async() => {
        if(mode === 'simple' && !description.trim()){
            toast.error("Please describe your song before creating.")
            return
        }

        if(mode === 'custom' && !styleInput.trim()){
            toast.error("Please add your preferred song style.")
            return
        }

        
        //Generate song

        let requestBody : GenerateRequest;

        if(mode === "simple"){
            requestBody = {
                fullDescribedSong: description,
                instrumental
            };
        }

        else{
            const prompt = styleInput;
            if(lyricsMode == "write"){
                requestBody= {
                    prompt,
                    lyrics,
                    instrumental
                }
            }
            else{
                requestBody ={
                    prompt,
                    describedLyrics: lyrics,
                    instrumental
                };
            }
        }

        try {

            setLoading(true);
            await generateSong(requestBody);
            router.refresh();
            setDescription("");
            setLyrics("");
            setStyleInput("");
            
        } catch (error) {

            toast.error("Failed to generate song");

        }finally{
            setLoading(false)
        }
        
    };

   return (
    <div className="bg-muted/30 flex w-full flex-col border-r lg:w-80">
        
        <Tabs value={mode} onValueChange={(value) => setMode(value as "simple" | "custom")} className="flex flex-col h-full">
            <div className="p-4 pb-0">
                <TabsList className="w-full">
                    <TabsTrigger value="simple">Simple</TabsTrigger>
                    <TabsTrigger value="custom">Custom</TabsTrigger>
                </TabsList>
            </div>

            <div className="flex-1 overflow-y-auto [scrollbar-width:none] [&::-webkit-scrollbar]:hidden p-4">
                <TabsContent value="simple" className="mt-2 space-y-6">
                    <div className="flex flex-col gap-3">
                        <label className="text-sm font-semibold">Describe your song</label>
                        <Textarea
                            placeholder="Eg: A dreamy lofi hip hop song, perfect for studying and relaxing..."
                            className="min-h-30 resize-none"
                            value={description}
                            onChange={(e) => setDescription(e.target.value)}
                        />
                    </div>

                    <div className="flex items-center justify-between">
                        <Button variant="outline" size="sm" onClick={() => setMode("custom")}><Plus className="mr-2"/>Add Lyrics</Button>
                        <div className="flex items-center gap-2">
                            <label className="text-sm font-medium">Instrumental</label>
                            <Switch checked={instrumental} onCheckedChange={setInstrumental} />
                        </div>
                    </div>

                    <div className="flex flex-col gap-3">
                        <label className="text-sm font-semibold">Popular Categories</label>
                        <div className="w-full overflow-x-auto whitespace-nowrap [scrollbar-width:none] [&::-webkit-scrollbar]:hidden">
                            <div className="flex gap-2 pb-2">
                                {inspirationTags.map((tag) => {
                                    const isSelected = description
                                        .split(", ")
                                        .map((s) => s.trim())
                                        .includes(tag);

                                    return (
                                        <button
                                            key={tag}
                                            onClick={() => handleInspirationTagClick(tag)}
                                            className={`flex items-center gap-1 rounded-full border px-3 py-1 text-xs transition-all cursor-pointer shrink-0
                                                ${isSelected
                                                    ? "border-primary bg-primary/20 text-primary"
                                                    : "border-border bg-transparent text-muted-foreground hover:border-primary/50 hover:text-foreground"
                                                }`}
                                        >
                                            <Music2Icon className="h-3 w-3" />
                                            {tag}
                                            {isSelected && <X className="h-3 w-3 ml-1" />}
                                        </button>
                                    );
                                })}
                            </div>
                        </div>
                    </div>
                </TabsContent>

                <TabsContent value="custom" className="flex-1 overflow-y-auto [scrollbar-width:none] [&::-webkit-scrollbar]:hidden p-4">
                     <div className="flex flex-col gap-3">
                        <div className="flex items-center justify-between">
                            <label className="text-sm font-semibold">Lyrics</label>
                            <div className="flex items-center gap-1">
                                <Button
                                variant={lyricsMode === "auto" ? "secondary" : "ghost"}
                                size="sm"
                                className="h-7 text-xs cursor-pointer"
                                onClick={() => {
                                setLyricsMode("auto");
                                setLyrics("");
                                }}
                                >
                                    Auto
                                </Button>

                                <Button
                                variant={lyricsMode === "write" ? "secondary" : "ghost"}
                                size="sm"
                                className="h-7 text-xs cursor-pointer"
                                onClick={() => {
                                setLyricsMode("write");
                                setLyrics("");
                                }}
                                >
                                    Write Lyrics
                                </Button>
                            </div>
                        </div>

                        <Textarea
                            placeholder={
                            lyricsMode === "write"
                                ? "Write your own lyrics here..."
                                : "Describe your lyrics, e.g., a sad song about lost love..."
                            }
                            value={lyrics}
                            onChange={(e) => setLyrics(e.target.value)}
                            className="min-h-25 resize-none"
                        />
                     </div>

                      <div className="mt-4 flex items-center justify-between">
                        <label className="text-sm font-medium">Instrumental</label>
                        <Switch
                            checked={instrumental}
                            onCheckedChange={setInstrumental}
                        />
                    </div>

                    {/* Styles */}

                    <div className="flex flex-col gap-3 mt-6">
                        <label className="text-sm font-semibold">Song Styles</label>

                        <Textarea
                            placeholder="Select some song style tags..."
                            value={styleInput}
                            onChange={(e) => setStyleInput(e.target.value)}
                            className="min-h-[60px] resize-none"
                        />

                        <div className="w-2xl overflow-x-auto whitespace-nowrap [scrollbar-width:none] [&::-webkit-scrollbar]:hidden mt-3">
                            <div className="flex gap-1.5 py-2">
                                {styleTags.map((tag) => {
                                    const isSelected = styleInput
                                        .split(", ")
                                        .map((s) => s.trim())
                                        .includes(tag);

                                    return (
                                        <span
                                            key={tag}
                                            onClick={() => handleStyleInputTagClick(tag)}
                                            className={`inline-flex items-center gap-1 rounded-md px-2.5 py-1 text-[11px] font-medium transition-all cursor-pointer shrink-0
                                                ${isSelected
                                                    ? "bg-amber-400/20 text-amber-300 ring-1 ring-amber-400/50"
                                                    : "bg-muted text-muted-foreground hover:bg-muted/80 hover:text-foreground"
                                                }`}
                                        >
                                            <span className="text-[10px] opacity-60">#</span>
                                            {tag}
                                            {isSelected && <X className="h-2.5 w-2.5 ml-0.5 opacity-70" />}
                                        </span>
                                    );
                                })}
                            </div>
                        </div>


                    </div>

                </TabsContent>
            </div>
        </Tabs>

        <div className="border-t p-4">
            <Button 
                onClick={handleCreate}
                disabled = {loading}
                className="w-full cursor-pointer bg-gradient-to-r from-violet-600 to-pink-600 hover:from-violet-700 hover:to-pink-700 text-white font-semibold shadow-lg shadow-violet-900/30 transition-all"
                size="lg"
            >   

                {loading ? <Loader2Icon className="mr-2 h-4 w-4 animate-spin" /> : <Music2Icon className="mr-2 h-4 w-4" />}
                {loading ? "Creating ..." : "Create Song"}
                
            </Button>
        </div>
    </div>

)
}