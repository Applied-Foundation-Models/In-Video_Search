import Link from 'next/link';
import { FaUser } from 'react-icons/fa'; // Import the account/user icon
import SigninButton from '../components/SigninButton';
import { Button } from '../components/ui/button';


const Header = () => {
  return (
<header className='flex h-14 items-center justify-between bg-background px-4 lg:px-6'>
        <Link href='#' className='flex items-center' prefetch={false}>
          <BookIcon className='mr-2 h-6 w-6' />
          <span className='text-lg font-semibold'>Educa</span>
        </Link>
        <nav className='hidden gap-6 lg:flex'>
          <Link
            href='/'
            className='text-sm font-medium underline-offset-4 hover:underline'
            prefetch={false}
          >
            Home
          </Link>
          <Link
            href='/about'
            className='text-sm font-medium underline-offset-4 hover:underline'
            prefetch={false}
          >
            About
          </Link>
          <Link
            href='/embed'
            className='text-sm font-medium underline-offset-4 hover:underline'
            prefetch={false}
          >
           Let&apos;s Learn
          </Link>
         <SigninButton />
         <FaUser className='mr-2 h-5 w-5' /> {/* Insert the account icon here */}
        </nav>
        <Button variant='outline' size='icon' className='lg:hidden'>
          <MenuIcon className='h-6 w-6' />
          <span className='sr-only'>Toggle navigation</span>
        </Button>
      </header>
  );
};
function BookIcon(props) {
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

function MenuIcon(props) {
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

export default Header;
