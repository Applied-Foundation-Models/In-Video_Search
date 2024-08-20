'use client'
import { signIn, signOut, useSession } from 'next-auth/react'
import React from 'react'
const SigninButton = () => {
  const { data: session } = useSession()

  if (session && session.user) {
    return (
      <div className='ml-auto flex gap-4'>
        <p className='text-sky-600'>{session.user.name}</p>
        <button onClick={() => signOut()} className='text-red-600'>
          Sign Out
        </button>
      </div>
    )
  }
  return (
    <button onClick={() => signIn()} className='ml-auto text-green-600'>
      Sign In
    </button>
  )
}

export default SigninButton
