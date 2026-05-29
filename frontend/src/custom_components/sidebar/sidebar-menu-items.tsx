"use client";

import { SidebarMenuButton, SidebarMenuItem } from "@/components/ui/sidebar";
import { Home, Music } from "lucide-react";
import { usePathname } from "next/navigation";

export default function SidebarMenuItems() {
  const path = usePathname();

  let items = [
    {
      title: "Home",
      url: "/",
      icon: Home,
      active: false,
    },
    {
      title: "Create",
      url: "/create",
      icon: Music,
      active: false,
    },
  ];

  items = items.map((item) => ({
    ...item,
    active: path === item.url,
  }));

  return (
    <>
      {items.map((item) => (
        <SidebarMenuItem key={item.title}>
          <SidebarMenuButton
            isActive={item.active}
            className={`
              group flex items-center gap-3 rounded-lg px-3 py-4 text-sm font-medium
              transition-all duration-200 ease-in-out
              ${
                item.active
                  ? "bg-primary/20 text-primary shadow-[inset_0_0_0_1px_rgba(139,92,246,0.3)]"
                  : "text-muted-foreground hover:bg-white/5 hover:text-foreground"
              }
            `}
          >
            <a href={item.url} className="flex items-center gap-3 w-full">
              <span
                className={`
                  flex h-8 w-8 items-center justify-center rounded-md shrink-0
                  transition-all duration-200
                  ${
                    item.active
                      ? "bg-primary/30 text-primary"
                      : "bg-white/5 text-muted-foreground group-hover:bg-white/10 group-hover:text-foreground"
                  }
                `}
              >
                <item.icon size={16} strokeWidth={2} />
              </span>
              <span className="tracking-wide">{item.title}</span>
              {item.active && (
                <span className="ml-auto h-1.5 w-1.5 rounded-full bg-primary" />
              )}
            </a>
          </SidebarMenuButton>
        </SidebarMenuItem>
      ))}
    </>
  );
}