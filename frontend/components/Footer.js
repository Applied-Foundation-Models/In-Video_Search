import Link from 'next/link';

// components/Footer.tsx
const Footer = () => {
  return (
      <footer className='w-full bg-muted p-6 md:py-12'>
        <div className='container grid max-w-7xl grid-cols-2 gap-8 text-sm sm:grid-cols-3 md:grid-cols-5'>
          <div className='grid gap-1'>
            <h3 className='font-semibold'>Company</h3>
            <Link href='/about' prefetch={false}>
              About Us
            </Link>
            <Link href='#' prefetch={false}>
              Our Team
            </Link>
            <Link href='#' prefetch={false}>
              Careers
            </Link>
            <Link href='#' prefetch={false}>
              News
            </Link>
          </div>
          <div className='grid gap-1'>
            <h3 className='font-semibold'>Support</h3>
            <Link href='#' prefetch={false}>
              Help Center
            </Link>
            <Link href='#' prefetch={false}>
              Contact Us
            </Link>
            <Link href='#' prefetch={false}>
              FAQs
            </Link>
            <Link href='#' prefetch={false}>
              Feedback
            </Link>
          </div>
          <div className='grid gap-1'>
            <h3 className='font-semibold'>Legal</h3>
            <Link href='#' prefetch={false}>
              Privacy Policy
            </Link>
            <Link href='#' prefetch={false}>
              Terms of Service
            </Link>
            <Link href='#' prefetch={false}>
              Cookie Policy
            </Link>
          </div>
              </div>
      </footer>
  );
};

export default Footer;
