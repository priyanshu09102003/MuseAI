import { getPresignedUrl } from '@/actions/generation'
import { SongCard } from '@/custom_components/song-card'
import { auth } from '@/lib/auth'
import { prisma } from '@/lib/prisma'
import { Music } from 'lucide-react'
import { headers } from 'next/headers'
import { redirect } from 'next/navigation'



export default async function MainPage () {
  const session = await auth.api.getSession({
    headers: await headers()
  })

  if(!session){
    redirect("/auth/sign-in")
  }

  const songs = await prisma.song.findMany({
    where: {
      published: true,
    },
    include: {
      user: {
        select: {
          name: true,
        },
      },
      _count: {
        select: {
          likes: true,
        },
      },
      categories: true,
      likes: session.user.id
        ? {
            where: {
              userId: session.user.id,
            },
          }
        : false,
    },
    orderBy: {
      createdAt: "desc",
    },
    take: 100,
  });

  const songsWithUrls = await Promise.all(
    songs.map(async (song) => {
      const thumbnailUrl = song.thumbnailS3Key
        ? await getPresignedUrl(song.thumbnailS3Key)
        : null;
      return { ...song, thumbnailUrl };
    }),
  );

  const currentYear = new Date().getFullYear();

  const trending   = songsWithUrls.slice(0, 10);
  const crazyHits  = songsWithUrls.slice(10, 20);
  const onRepeat   = songsWithUrls.slice(20, 30);
  const deepCuts   = songsWithUrls.slice(30, 40);
  const bangers    = songsWithUrls.slice(40, 50);

  const sections = [
    { heading: `🔥 Best of ${currentYear}`, songs: trending },
    { heading: "⚡ Crazy Hits",             songs: crazyHits },
    { heading: "🎧 On Repeat",              songs: onRepeat },
    { heading: "🌊 Deep Cuts",              songs: deepCuts },
    { heading: "💥 Absolute Bangers",       songs: bangers },
  ].filter(({ songs }) => songs.length > 0);

  if (sections.length === 0) {
    return (
      <div className="flex h-full flex-col items-center justify-center p-4 text-center">
        <Music className="text-muted-foreground h-20 w-20" />
        <h1 className="mt-4 text-2xl font-bold tracking-tight">
          No Music Here
        </h1>
        <p className="text-muted-foreground mt-2">
          There are no published songs available right now. Check back later!
        </p>
      </div>
    );
  }

  return (
    <div className="p-4">
      <h1 className="text-3xl font-bold tracking-tight">Discover Music</h1>

      {sections.map(({ heading, songs }) => (
        <div key={heading} className="mt-6">
          <h2 className="text-xl font-semibold">{heading}</h2>
          <div className="mt-4 grid grid-cols-2 gap-x-4 gap-y-6 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 xl:grid-cols-6">
            {songs.map((song) => (
              <SongCard key={song.id} song={song} />
            ))}
          </div>
        </div>
      ))}
    </div>
  );
}