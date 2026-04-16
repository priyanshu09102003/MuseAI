import React from 'react'
import { type Metadata } from "next";
import { Geist } from "next/font/google";
import "../globals.css";

export const metadata: Metadata = {
  title: "MuseAI | Home",
  description: "Music Generator",
  icons: [{ rel: "icon", url: "/favicon.ico" }],
};

const geist = Geist({
  subsets: ["latin"],
  variable: "--font-geist-sans",
});

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {

    return(
        <html lang="en" className={`${geist.variable}`}>
            <body>
                {children}
            </body>
        </html>
    )
}
