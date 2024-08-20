'use client'
import { Button } from '@/components/ui/button'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger
} from '@/components/ui/dropdown-menu'
import { Input } from '@/components/ui/input'
import { Textarea } from '@/components/ui/textarea'
import { useEffect, useState } from 'react'

type Lecture = {
  id: number
  title: string
}

export function SearchFunctionality() {
  // State to hold the list of available lectures (videos)
  const [lectures, setLectures] = useState<Lecture[]>([])

  // State to track the currently selected lecture
  const [selectedLecture, setSelectedLecture] = useState<Lecture | null>(null)

  // State to hold the URL of the video to be displayed in the player
  const [videoUrl, setVideoUrl] = useState<string>('')

  // State for handling search term input
  const [searchTerm, setSearchTerm] = useState<string>('')

  // State to store the search result for the current frame summary
  const [searchResult, setSearchResult] = useState<string | null>(null)

  // useEffect hook to fetch the list of lectures (videos) from the backend when the component first mounts
  useEffect(() => {
    // Function to fetch the lectures
    const fetchLectures = async () => {
      // Fetching the lectures from the backend
      const response = await fetch('http://localhost:5000/videos')
      const data = await response.json()

      // Sorting the lectures by ID (you can change this to sort by other criteria like title)
      const sortedLectures = data.sort((a: Lecture, b: Lecture) => a.id - b.id)

      // Updating the lectures state with the sorted list
      setLectures(sortedLectures)

      // Set the default video URL to the first lecture in the sorted list
      if (sortedLectures.length > 0) {
        const firstLecture = sortedLectures[0] // The first lecture in the sorted list
        setSelectedLecture(firstLecture) // Set this lecture as the selected one
        setVideoUrl(`http://localhost:5000/data/raw/${firstLecture.id}.mp4`) // Set the video URL for this lecture
      }
    }

    // Call the fetchLectures function
    fetchLectures()
  }, []) // Empty dependency array means this useEffect runs only once on mount

  // Function to handle when a lecture is selected from the dropdown
  const handleLectureSelect = (lecture: Lecture) => {
    setSelectedLecture(lecture) // Set the selected lecture
    setVideoUrl(`http://localhost:5000/data/raw/${lecture.id}.mp4`) // Update the video URL based on the selected lecture ID
  }

  // Function to handle the search operation for frame summaries
  const handleSearch = () => {
    // Mock data for frame summaries based on lecture IDs
    const summaries: { [key: number]: string } = {
      1: 'This frame discusses the basics of React components.',
      2: 'This frame explains how to manage state in a React application.',
      3: 'This frame introduces the concept of React Hooks.',
      4: 'This frame covers the fundamentals of React Router.',
      5: 'This frame talks about techniques to optimize React performance.'
    }

    // Find the summary for the selected lecture based on the search term
    const result = summaries[selectedLecture?.id || 0]

    // Update the search result state with the found summary
    setSearchResult(result)

    // Move the video to a specific time based on the selected lecture (just a mock example here)
    const video = document.querySelector('video')
    if (video) {
      video.currentTime = (selectedLecture?.id || 0) * 10 // Example: move to 10 seconds for lecture ID 1, 20 seconds for ID 2, etc.
    }
  }

  return (
    <div className='container mx-auto max-w-5xl px-4 py-8'>
      <div className='grid gap-6'>
        <div className='flex items-center gap-4'>
          {/* Dropdown menu for selecting lectures */}
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant='outline' className='w-full'>
                {selectedLecture ? selectedLecture.title : 'Select lectures'}
                <ChevronDownIcon className='ml-auto h-4 w-4' />
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent className='w-[300px]'>
              {/* Populate the dropdown with lectures */}
              {lectures.map(lecture => (
                <DropdownMenuItem
                  key={lecture.id}
                  onSelect={() => handleLectureSelect(lecture)} // Handle lecture selection
                  className={
                    selectedLecture?.id === lecture.id ? 'bg-muted' : ''
                  } // Highlight the selected lecture
                >
                  {lecture.title}
                </DropdownMenuItem>
              ))}
            </DropdownMenuContent>
          </DropdownMenu>

          {/* Input field for searching frame summaries */}
          <div className='relative w-full'>
            <Input
              type='text'
              placeholder='Search for a frame summary'
              value={searchTerm}
              onChange={e => setSearchTerm(e.target.value)} // Update search term state
              className='pr-12'
            />
            <Button
              variant='ghost'
              size='icon'
              className='absolute right-2 top-2'
              onClick={handleSearch} // Handle the search action
            >
              <SearchIcon className='h-5 w-5' />
            </Button>
          </div>
        </div>

        {/* Video player to display the selected video */}
        <div className='overflow-hidden rounded-lg'>
          <video src={videoUrl} controls className='aspect-video w-full' />
        </div>

        {/* Display the search result summary */}
        {searchResult && (
          <div className='rounded-lg bg-muted p-4'>
            <p>{searchResult}</p>
          </div>
        )}

        {/* Read-only text area to show the summary of the current frame */}
        <div className='rounded-lg bg-muted p-4'>
          <Textarea
            placeholder='Summary of the current frame'
            value={searchResult || ''}
            readOnly
            className='w-full'
          />
        </div>
      </div>
    </div>
  )
}

function ChevronDownIcon(props: React.SVGProps<SVGSVGElement>) {
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
      <path d='m6 9 6 6 6-6' />
    </svg>
  )
}

function SearchIcon(props: React.SVGProps<SVGSVGElement>) {
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
      <circle cx='11' cy='11' r='8' />
      <path d='m21 21-4.3-4.3' />
    </svg>
  )
}

export default SearchFunctionality
