import {
  Sidebar,
  SidebarContent,
  SidebarFooter,
  SidebarGroup,
  SidebarGroupContent,
  SidebarGroupLabel,
  SidebarMenu,
} from "@/components/ui/sidebar";

import SidebarMenuItems from "./sidebar-menu-items";
import { Credits } from "./credits";
import { UserButton } from "@daveyplate/better-auth-ui";
import { User } from "lucide-react";
import Upgrade from "./upgrade";


export async function AppSidebar() {
  return (
    <Sidebar>
      <SidebarContent>
        <SidebarGroup>
          {/* Logo */}
          <SidebarGroupLabel className="mt-5 mb-8 px-3">
            <div className="flex items-center gap-3">
              {/* Icon mark */}
              <div className="relative flex h-9 w-9 items-center justify-center rounded-xl bg-primary/20 shadow-[0_0_12px_rgba(139,92,246,0.3)] ring-1 ring-primary/30">
                <svg
                  width="18"
                  height="18"
                  viewBox="0 0 24 24"
                  fill="none"
                  className="text-primary"
                >
                  <path
                    d="M9 18V5l12-2v13"
                    stroke="currentColor"
                    strokeWidth="2"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  />
                  <circle cx="6" cy="18" r="3" fill="currentColor" opacity="0.7" />
                  <circle cx="18" cy="16" r="3" fill="currentColor" />
                </svg>
              </div>
              {/* Wordmark */}
              <div className="flex flex-col leading-none">
                <span className="text-xl font-black tracking-widest text-foreground">
                  Muse
                  <span className="text-primary">AI</span>
                </span>
                <span className="text-[10px] font-medium tracking-[0.2em] text-muted-foreground uppercase">
                  AI Music Studio
                </span>
              </div>
            </div>
          </SidebarGroupLabel>

          {/* Nav section label */}
          <div className="px-3 mb-2">
            <span className="text-[10px] font-semibold uppercase tracking-[0.15em] text-muted-foreground/60">
              Navigation
            </span>
          </div>

          <SidebarGroupContent>
            <SidebarMenu className="gap-3.5">
              <SidebarMenuItems />
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>
      </SidebarContent>
      <SidebarFooter>
        <div className="mb-2 flex w-full items-center justify-center gap-1 text-xs">
          <Credits />
          <Upgrade />
        </div>
        <UserButton
        variant={"outline"}
        additionalLinks={[
          {
            label: "Customer Portal",
            href: "/customer-portal",
            icon: <User/>
          },
          
        ]}
        />

      </SidebarFooter>
    </Sidebar>
  );
}