import { useState } from 'react'
import reactLogo from './assets/react.svg'
import viteLogo from '/vite.svg'
import './App.css'

function App() {
  const [count, setCount] = useState(0)

  return (
    <>
      <div>
        <h1>Hello, World</h1>
        <p> This simple React app is part of the technical challenge for ML+AI Software Development under Dr.Lence!</p>
      </div>
    </>
  )
}

export default App
