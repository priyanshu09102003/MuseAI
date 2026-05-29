
import { SongPanel } from '@/custom_components/createPage/song-panel'
import TrackListFetcher from '@/custom_components/createPage/track-list-fetcher'
import { auth } from '@/lib/auth'
import { Loader2Icon } from 'lucide-react'
import { headers } from 'next/headers'
import { redirect } from 'next/navigation'
import { Suspense } from 'react'

export default async function MainPage () {
  const session = await auth.api.getSession({
    headers: await headers()
  })

  if(!session){
    redirect("/auth/sign-in")
  }
  return (
    <div className='flex h-full flex-col lg:flex-row overflow-hidden'>
        <SongPanel />
        <Suspense fallback={<div className='flex h-full w-full items-center justify-center'><Loader2Icon className='h-8 w-8 animate-spin' /></div>}>
            <TrackListFetcher />
        </Suspense>
    </div>
  )
}


