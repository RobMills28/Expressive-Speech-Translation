import React, { useState } from 'react';

const LandingPage = () => {
  const [view, setView] = useState('landing'); // landing, login, signup

  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-50 to-white">
      {/* Navigation */}
      <nav className="border-b bg-white">
        <div className="max-w-6xl mx-auto px-4 py-4 flex items-center justify-between">
          <h1 className="text-xl font-bold text-purple-600">Magenta</h1>
          <div className="flex items-center gap-4">
            <button 
              onClick={() => setView('login')}
              className="px-4 py-2 text-gray-600 hover:text-gray-900 transition-colors"
            >
              Log in
            </button>
            <button 
              onClick={() => setView('signup')}
              className="px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 transition-colors"
            >
              Start Creating
            </button>
          </div>
        </div>
      </nav>

      {/* Hero Section */}
      <div className="max-w-6xl mx-auto px-4">
        <div className="text-center max-w-3xl mx-auto pt-32">
          <h1 className="text-5xl font-bold mb-6 bg-clip-text text-transparent bg-gradient-to-r from-purple-600 to-purple-900">
            Share Your Content With the World, In Any Language
          </h1>
          <p className="text-xl text-gray-600 mb-8">
            Magenta helps creators reach global audiences by translating content while 
            preserving their authentic voice and style.
          </p>
          <button 
            onClick={() => setView('signup')}
            className="px-8 py-4 bg-purple-600 text-white rounded-lg hover:bg-purple-700 text-lg shadow-lg hover:shadow-xl transition-all duration-200"
          >
            Get Started Free
          </button>
        </div>

        {/* Visual Elements */}
        <div className="pointer-events-none absolute inset-0 -z-10 overflow-hidden">
          <div className="absolute -top-1/2 left-1/2 -translate-x-1/2 w-full max-w-4xl h-[800px] rounded-full bg-purple-100/50 blur-3xl" />
          <div className="absolute top-1/4 right-1/4 w-[600px] h-[600px] rounded-full bg-fuchsia-100/30 blur-3xl" />
        </div>
      </div>

      {/* Placeholder for illustrations or demo */}
      <div className="max-w-6xl mx-auto mt-16 px-4">
        <div className="aspect-video w-full max-w-3xl mx-auto rounded-xl bg-white/50 shadow-lg backdrop-blur-sm border border-white/20">
          {/* Platform preview/demo would go here */}
        </div>
      </div>
    </div>
  );
};

export default LandingPage;