import Image from 'next/image'
import Link from 'next/link'
import React from 'react'

// Main landing page component
const LandingPage: React.FC = () => {
  return (
    <div className='flex min-h-[100dvh] flex-col'>
      <main className='flex-1'>
        <section className='w-full py-12 md:py-24 lg:py-32'>
          <div className='container px-4 md:px-6'>
            <div className='grid gap-6 lg:grid-cols-2 lg:gap-12'>
              <div className='flex flex-col justify-center space-y-4'>
                <div className='space-y-2'>
                  <h1 className='text-3xl font-bold tracking-tighter sm:text-4xl md:text-5xl lg:text-6xl/none'>
                    Learn Anytime, Anywhere
                  </h1>
                  <p className='max-w-[600px] text-muted-foreground md:text-xl'>
                    Our education app provides a seamless learning experience
                    for students of all ages. Explore a wide range of
                    interactive courses and resources.
                  </p>
                </div>
                <div className='flex flex-col gap-2 min-[400px]:flex-row'>
                  <Link
                    href='#'
                    className='inline-flex h-10 items-center justify-center rounded-md bg-primary px-8 text-sm font-medium text-primary-foreground shadow transition-colors hover:bg-primary/90 focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring disabled:pointer-events-none disabled:opacity-50'
                    prefetch={false}
                  >
                    <p>Let&apos;s Learn</p>
                  </Link>
                  <Link
                    href='#'
                    className='inline-flex h-10 items-center justify-center rounded-md border border-input bg-background px-8 text-sm font-medium shadow-sm transition-colors hover:bg-accent hover:text-accent-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring disabled:pointer-events-none disabled:opacity-50'
                    prefetch={false}
                  >
                    About Us
                  </Link>
                </div>
              </div>
              <Image
                src='/placeholder.svg'
                width={550}
                height={550}
                alt='Hero'
                className='mx-auto aspect-video overflow-hidden rounded-xl object-cover sm:w-full lg:aspect-square'
              />
            </div>
          </div>
        </section>
        <section className='w-full bg-muted py-12 md:py-24 lg:py-32'>
          <div className='container px-4 md:px-6'>
            <div className='grid gap-6 lg:grid-cols-2 lg:gap-12'>
              <div>
                <div className='aspect-video overflow-hidden rounded-xl'>
                  <div />
                </div>
              </div>
              <div className='flex flex-col justify-center space-y-4'>
                <div className='space-y-2'>
                  <h2 className='text-3xl font-bold tracking-tighter md:text-4xl/tight'>
                    Discover Our App
                  </h2>
                  <p className='max-w-[600px] text-muted-foreground md:text-xl/relaxed lg:text-base/relaxed xl:text-xl/relaxed'>
                    Watch our video to learn how our education app can help you
                    or your students achieve their learning goals. Explore our
                    wide range of interactive courses and resources.
                  </p>
                </div>
              </div>
            </div>
          </div>
        </section>
      </main>
    </div>
  )
}

// BookIcon component
function BookIcon(props: React.SVGProps<SVGSVGElement>) {
  return (
    <svg
      {...props}
      xmlns='http://www.w3.org/2000/svg'
      width='24'
      height='24'
      viewBox='0 0 24 24'
      fill='none'
      stroke='currentColor'
      strokeWidth='2'
      strokeLinecap='round'
      strokeLinejoin='round'
    >
      <path d='M4 19.5v-15A2.5 2.5 0 0 1 6.5 2H20v20H6.5a2.5 2.5 0 0 1 0-5H20' />
    </svg>
  )
}

// MenuIcon component
function MenuIcon(props: React.SVGProps<SVGSVGElement>) {
  return (
    <svg
      {...props}
      xmlns='http://www.w3.org/2000/svg'
      width='24'
      height='24'
      viewBox='0 0 24 24'
      fill='none'
      stroke='currentColor'
      strokeWidth='2'
      strokeLinecap='round'
      strokeLinejoin='round'
    >
      <line x1='4' x2='20' y1='12' y2='12' />
      <line x1='4' x2='20' y1='6' y2='6' />
      <line x1='4' x2='20' y1='18' y2='18' />
    </svg>
  )
}

export default LandingPage
