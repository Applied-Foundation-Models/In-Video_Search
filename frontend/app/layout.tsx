// This is the root layout component for your Next.js app.
// Learn more: https://nextjs.org/docs/app/building-your-application/routing/pages-and-layouts#root-layout-required
import { cn } from '@/lib/utils'
import { Inter } from 'next/font/google'
import Footer from '../components/Footer'
import Header from '../components/Header'
import Providers from '../components/Providers'

import { ReactNode } from 'react' // Import ReactNode from React
import './globals.css'

const fontHeading = Inter({
  subsets: ['latin'],
  display: 'swap',
  variable: '--font-heading'
})

const fontBody = Inter({
  subsets: ['latin'],
  display: 'swap',
  variable: '--font-body'
})

type LayoutProps = {
  children: ReactNode // Define the type for children
}

export default function Layout({ children }: LayoutProps) {
  return (
    <html lang='en'>
      <body
        className={cn('antialiased', fontHeading.variable, fontBody.variable)}
      >
        <Providers>
          <Header />
          {children}
        </Providers>
        <Footer />
      </body>
    </html>
  )
}
