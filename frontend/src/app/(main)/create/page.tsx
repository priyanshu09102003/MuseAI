import CreateSong from '@/custom_components/create'
import { SongPanel } from '@/custom_components/createPage/song-panel'
import { auth } from '@/lib/auth'
import { headers } from 'next/headers'
import { redirect } from 'next/navigation'
import React from 'react'

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
    </div>
  )
}


