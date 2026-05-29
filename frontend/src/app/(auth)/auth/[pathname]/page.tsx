// app/auth/[pathname]/page.tsx
import { authViewPaths } from "@daveyplate/better-auth-ui/server";

import { MusicAuthShell } from "@/components/auth-shell"
import { AuthCard } from "@/components/auth";

export function generateStaticParams() {
  return Object.values(authViewPaths).map((pathname) => ({ pathname }));
}

export default async function AuthPage({
  params,
}: {
  params: Promise<{ pathname: string }>;
}) {
  const { pathname } = await params;

  return (
    <MusicAuthShell>
      <AuthCard pathname={pathname} />
    </MusicAuthShell>
  );
}