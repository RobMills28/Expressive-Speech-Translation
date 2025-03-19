import React from 'react';
import { BrowserRouter as Router, Routes, Route, Link, useLocation } from 'react-router-dom';
import { Card } from "./components/ui/card";
import { Button } from "./components/ui/button";
import { Globe, Youtube, Music, Video } from 'lucide-react';
import Dashboard from "./components/Dashboard";
import PricingPage from './components/PricingPage';

// Navigation Component
const Navigation = () => {
  const location = useLocation();

  // Don't show navigation on landing page
  if (location.pathname === '/') {
    return null;
  }
  
  return (
    <nav className="bg-white border-b">
      <div className={`mx-auto px-8 ${location.pathname === '/creator-studio' ? 'max-w-full' : 'max-w-6xl'}`}>
        <div className="flex items-center justify-between h-16">
          <div className="flex items-center">
            <Link to="/" className="text-xl font-bold text-fuchsia-600">
              Magenta AI
            </Link>
          </div>
          
          <div className="flex items-center gap-12">
            <Link 
              to="/creator-studio"
              className={`px-4 py-2 rounded-lg hover:bg-fuchsia-700 ${
                location.pathname === '/creator-studio' 
                  ? 'bg-fuchsia-700 text-white'
                  : 'bg-fuchsia-600 text-white'
              }`}
            >
              Creator Studio
            </Link>
          </div>
        </div>
      </div>
    </nav>
  );
};

// Landing Page Component
const LandingPage = () => {
  return (
    <div className="relative min-h-screen">
      <div className="min-h-screen bg-gradient-to-br from-fuchsia-50 to-white">
        <nav className="border-b bg-white">
          <div className="max-w-6xl mx-auto px-4 py-4 flex items-center justify-between">
            <h1 className="text-xl font-bold text-fuchsia-600">Magenta AI</h1>
            <div className="flex items-center gap-4">
            <Link
                to="/pricing"
                className="px-4 py-2 text-gray-600 hover:text-gray-900 transition-colors"
              >
                Pricing
              </Link>
              <Link
                to="/creator-studio"
                className="px-4 py-2 bg-fuchsia-600 text-white rounded-lg hover:bg-fuchsia-700 transition-colors"
              >
                Creator Studio
              </Link>
            </div>
          </div>
        </nav>

        <div className="max-w-6xl mx-auto px-4">
          <div className="text-center max-w-3xl mx-auto pt-32">
            <h1 className="text-5xl font-bold mb-6 bg-clip-text text-transparent bg-gradient-to-r from-fuchsia-600 to-pink-600 leading-tight">
              Share Your Content With The World, In Any Language
            </h1>
            <p className="text-xl text-gray-600 mb-8">
              Magenta helps creators reach global audiences by translating content while 
              preserving their authentic voice and style.
            </p>
            <Link
              to="/creator-studio"
              className="inline-block px-8 py-4 bg-fuchsia-600 text-white rounded-lg hover:bg-fuchsia-700 text-lg shadow-lg hover:shadow-xl transition-all duration-200"
            >
              Get Started Free
            </Link>
          </div>

          {/* Content Creator Section - Side by Side */}
          <div className="max-w-6xl mx-auto mt-32 px-4">
            <div className="flex flex-col lg:flex-row gap-8 items-center">
              
              {/* Left side - Video */}
              <div className="lg:w-1/2">
                <div className="aspect-video w-full rounded-xl bg-white/50 shadow-lg backdrop-blur-sm border border-white/20 overflow-hidden">
                  <video 
                    className="w-full h-full object-cover"
                    autoPlay
                    muted
                    loop
                    playsInline
                    src={`${process.env.PUBLIC_URL}/videos/man-speaking-into-the-microphone.mp4`}
                  >
                    Your browser does not support the video tag.
                  </video>
                </div>
              </div>
              
              {/* Right side - Content */}
              <div className="lg:w-1/2">
                <h2 className="text-3xl font-bold mb-6 text-fuchsia-800">Translate Your Content While Preserving Your Unique Style</h2>
                <p className="text-lg text-gray-700 mb-6">
                  Maintain your authentic voice, emotions, and delivery when translating your videos and podcasts to new languages. Our AI preserves the nuances that make your content special.
                </p>
                <div className="space-y-4">
                  <div className="flex items-start">
                    <div className="mt-1 bg-fuchsia-100 p-2 rounded-full">
                      <Globe className="w-5 h-5 text-fuchsia-600" />
                    </div>
                    <div className="ml-4">
                      <h3 className="text-lg font-semibold">Reach Global Audiences</h3>
                      <p className="text-gray-600">Break language barriers and connect with viewers from around the world.</p>
                    </div>
                  </div>
                  <div className="flex items-start">
                    <div className="mt-1 bg-fuchsia-100 p-2 rounded-full">
                      <Music className="w-5 h-5 text-fuchsia-600" />
                    </div>
                    <div className="ml-4">
                      <h3 className="text-lg font-semibold">Emotional Preservation</h3>
                      <p className="text-gray-600">Keep your vocal characteristics, tone, and emotional delivery intact.</p>
                    </div>
                  </div>
                  <div className="flex items-start">
                    <div className="mt-1 bg-fuchsia-100 p-2 rounded-full">
                      <Video className="w-5 h-5 text-fuchsia-600" />
                    </div>
                    <div className="ml-4">
                      <h3 className="text-lg font-semibold">Multi-Platform Ready</h3>
                      <p className="text-gray-600">Expand your reach on YouTube, TikTok, Spotify, and beyond with optimized translations.</p>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
          
          {/* Second Content Section - Reversed Layout with Image */}
          <div className="max-w-6xl mx-auto mt-36 px-4 mb-32">
            <div className="flex flex-col lg:flex-row-reverse gap-8 items-center">
              
              {/* Right side - Image */}
              <div className="lg:w-1/2">
                <div className="aspect-video w-full rounded-xl bg-white/50 shadow-lg backdrop-blur-sm border border-white/20 overflow-hidden">
                  <img 
                    className="w-full h-full object-cover"
                    src={`${process.env.PUBLIC_URL}/images/Woman-Recording-Herself-While-Cooking.jpg`}
                    alt="Content creator recording cooking video"
                  />
                </div>
              </div>
              
              {/* Left side - Content */}
              <div className="lg:w-1/2">
                <h2 className="text-3xl font-bold mb-6 text-fuchsia-800">Grow Your Global Presence</h2>
                <p className="text-lg text-gray-700 mb-6">
                  Unlock new audiences and opportunities by making your content accessible to viewers all over the world. Magenta AI's translation technology helps you scale your reach without scaling your effort.
                </p>
                <div className="space-y-4">
                  <div className="flex items-start">
                    <div className="mt-1 bg-fuchsia-100 p-2 rounded-full">
                      <Youtube className="w-5 h-5 text-fuchsia-600" />
                    </div>
                    <div className="ml-4">
                      <h3 className="text-lg font-semibold">Dominate Social Platforms</h3>
                      <p className="text-gray-600">Optimize for YouTube, TikTok, Instagram and more with platform-specific translations.</p>
                    </div>
                  </div>
                  <div className="flex items-start">
                    <div className="mt-1 bg-fuchsia-100 p-2 rounded-full">
                      <Globe className="w-5 h-5 text-fuchsia-600" />
                    </div>
                    <div className="ml-4">
                      <h3 className="text-lg font-semibold">Multiple Languages</h3>
                      <p className="text-gray-600">Translate into dozens of languages with a single click and minimal effort.</p>
                    </div>
                  </div>
                  <div className="flex items-start">
                    <div className="mt-1 bg-fuchsia-100 p-2 rounded-full">
                      <Music className="w-5 h-5 text-fuchsia-600" />
                    </div>
                    <div className="ml-4">
                      <h3 className="text-lg font-semibold">Authentic Audience Connection</h3>
                      <p className="text-gray-600">Connect with international audiences in their native language, building loyalty worldwide.</p>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>

          <div className="pointer-events-none absolute inset-0 -z-10 overflow-hidden">
            <div className="absolute -top-1/2 left-1/2 -translate-x-1/2 w-full max-w-4xl h-[800px] rounded-full bg-fuchsia-100/50 blur-3xl" />
            <div className="absolute top-1/4 right-1/4 w-[600px] h-[600px] rounded-full bg-pink-100/30 blur-3xl" />
          </div>
        </div>
      </div>
      
      <footer className="w-full bg-white border-t py-6 mt-20">
        <div className="max-w-6xl mx-auto px-4">
          <div className="flex flex-col md:flex-row justify-between items-center">
            <p className="text-gray-500 text-sm">© {new Date().getFullYear()} Magenta AI. All rights reserved.</p>
            <div className="flex space-x-6 mt-4 md:mt-0">
              <Link to="/terms" className="text-gray-500 hover:text-fuchsia-600 text-sm">
                Terms
              </Link>
              <Link to="/privacy" className="text-gray-500 hover:text-fuchsia-600 text-sm">
                Privacy
              </Link>
              <Link to="/contact" className="text-gray-500 hover:text-fuchsia-600 text-sm">
                Contact
              </Link>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
};

// Main App Component
const App = () => {
  return (
    <Router>
      <div className="min-h-screen bg-white">
        <Navigation />
        <Routes>
          <Route path="/" element={<LandingPage />} />
          <Route path="/pricing" element={<PricingPage />} />
          <Route path="/creator-studio" element={<Dashboard />} />
        </Routes>
      </div>
    </Router>
  );
};

export default App;