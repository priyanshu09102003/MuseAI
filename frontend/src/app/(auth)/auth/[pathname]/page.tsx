import { viewPaths } from "@better-auth-ui/react/core"
import { notFound } from "next/navigation"

import { Auth } from "@/components/auth"
import { MusicAuthShell } from "@/components/auth-shell"

export default async function AuthPage({
  params,
}: {
  params: Promise<{
    pathname: string
  }>
}) {
  const { pathname } = await params

  if (!Object.values(viewPaths.auth).includes(pathname)) {
    notFound()
  }

  return (
    <MusicAuthShell>
      <Auth path={pathname} />
    </MusicAuthShell>
  )
}