"use client"

import { BreadcrumbPage } from "@/components/ui/breadcrumb"
import { usePathname } from "next/navigation"

export default function BreadcrumbPageClient(){

    const path = usePathname();


    return (
        <BreadcrumbPage>
            {path === "/" && "Home"}
            {path === "/create" && "Create"}      
        </BreadcrumbPage>
    )

}