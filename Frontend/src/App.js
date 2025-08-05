import React from 'react';
import { BrowserRouter as Router, Routes, Route, Link, useLocation } from 'react-router-dom';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "./components/ui/card";
import { Button } from "./components/ui/button";
import { Badge } from "./components/ui/badge";
import { Globe, Youtube, Music, Video, AudioWaveform, Heart, Clock, Mic, Zap, Users, ArrowRight, Play, Volume2 } from 'lucide-react';
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
    <div className="min-h-screen bg-white">
      {/* Navigation */}
      <nav className="border-b border-gray-200 bg-white sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-4">
            <div className="flex items-center space-x-2">
              <div className="w-8 h-8 bg-fuchsia-600 rounded-lg flex items-center justify-center">
                <AudioWaveform className="w-5 h-5 text-white" />
              </div>
              <span className="text-xl font-semibold">Magenta AI</span>
            </div>
            <div className="hidden md:flex items-center space-x-8">
              <a href="#features" className="text-gray-600 hover:text-gray-900 transition-colors">Features</a>
              <a href="#technology" className="text-gray-600 hover:text-gray-900 transition-colors">Technology</a>
              <a href="#use-cases" className="text-gray-600 hover:text-gray-900 transition-colors">Use Cases</a>
              <Link to="/creator-studio">
                <Button variant="outline" size="sm">Get Started</Button>
              </Link>
            </div>
          </div>
        </div>
      </nav>

      {/* Hero Section */}
      <section className="relative py-20 lg:py-32">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center max-w-4xl mx-auto">
            <Badge variant="secondary" className="mb-6">
              🎤 Revolutionary Speech Translation
            </Badge>
            <h1 className="text-4xl md:text-6xl lg:text-7xl mb-6 font-bold">
              Share Your Content
              <br />
              <span className="text-fuchsia-600">With The World,</span>
              <br />
              In Any Language
            </h1>
            <p className="text-xl text-gray-600 mb-8 max-w-2xl mx-auto">
              Magenta helps creators reach global audiences by translating content while 
              preserving their authentic voice and style.
            </p>
            <div className="flex flex-col sm:flex-row gap-4 justify-center items-center">
              <Link to="/creator-studio">
                <Button size="lg" className="group bg-fuchsia-600 hover:bg-fuchsia-700">
                  Get Started Free
                  <Play className="w-4 h-4 ml-2 group-hover:translate-x-1 transition-transform" />
                </Button>
              </Link>
              <Button variant="outline" size="lg">
                Watch Demo
                <ArrowRight className="w-4 h-4 ml-2" />
              </Button>
            </div>
          </div>
        </div>
        
        {/* Visual Elements */}
        <div className="absolute inset-0 -z-10 overflow-hidden">
          <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-96 h-96 bg-fuchsia-100 rounded-full blur-3xl"></div>
          <div className="absolute top-1/4 right-1/4 w-64 h-64 bg-pink-100 rounded-full blur-2xl"></div>
        </div>
      </section>

      {/* Features Section */}
      <section id="features" className="py-20 bg-gray-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-16">
            <h2 className="text-3xl md:text-4xl mb-4 font-bold">
              Beyond Words: <span className="text-fuchsia-600">Preserve Everything</span>
            </h2>
            <p className="text-xl text-gray-600 max-w-3xl mx-auto">
              Our breakthrough technology maintains the full spectrum of human expression across languages
            </p>
          </div>
          
          <div className="grid md:grid-cols-3 gap-8">
            <Card className="border-0 bg-white shadow-sm">
              <CardHeader>
                <div className="w-12 h-12 bg-fuchsia-100 rounded-lg flex items-center justify-center mb-4">
                  <Heart className="w-6 h-6 text-fuchsia-600" />
                </div>
                <CardTitle>Emotional Fidelity</CardTitle>
                <CardDescription>
                  Preserve joy, sadness, excitement, and every nuance of emotion across languages
                </CardDescription>
              </CardHeader>
            </Card>
            
            <Card className="border-0 bg-white shadow-sm">
              <CardHeader>
                <div className="w-12 h-12 bg-fuchsia-100 rounded-lg flex items-center justify-center mb-4">
                  <Mic className="w-6 h-6 text-fuchsia-600" />
                </div>
                <CardTitle>Vocal Characteristics</CardTitle>
                <CardDescription>
                  Maintain pitch, tone, accent, and speaking style for authentic communication
                </CardDescription>
              </CardHeader>
            </Card>
            
            <Card className="border-0 bg-white shadow-sm">
              <CardHeader>
                <div className="w-12 h-12 bg-fuchsia-100 rounded-lg flex items-center justify-center mb-4">
                  <Clock className="w-6 h-6 text-fuchsia-600" />
                </div>
                <CardTitle>Natural Timing</CardTitle>
                <CardDescription>
                  Preserve pauses, rhythm, and conversational flow for natural interactions
                </CardDescription>
              </CardHeader>
            </Card>
          </div>
        </div>
      </section>

      {/* Technology Section */}
      <section id="technology" className="py-20">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="grid lg:grid-cols-2 gap-16 items-center">
            <div>
              <Badge variant="outline" className="mb-4">
                🔬 Advanced AI Technology
              </Badge>
              <h2 className="text-3xl md:text-4xl mb-6 font-bold">
                Translate Your Content While
                <br />
                <span className="text-fuchsia-600">Preserving Your Unique Style</span>
              </h2>
              <p className="text-lg text-gray-600 mb-8">
                Maintain your authentic voice, emotions, and delivery when translating your videos and podcasts to new languages. Our AI preserves the nuances that make your content special.
              </p>
              
              <div className="space-y-4">
                <div className="flex items-start space-x-3">
                  <div className="w-2 h-2 bg-fuchsia-600 rounded-full mt-2"></div>
                  <div>
                    <h4 className="font-medium mb-1">Reach Global Audiences</h4>
                    <p className="text-gray-600">Break language barriers and connect with viewers from around the world.</p>
                  </div>
                </div>
                <div className="flex items-start space-x-3">
                  <div className="w-2 h-2 bg-fuchsia-600 rounded-full mt-2"></div>
                  <div>
                    <h4 className="font-medium mb-1">Emotional Preservation</h4>
                    <p className="text-gray-600">Keep your vocal characteristics, tone, and emotional delivery intact.</p>
                  </div>
                </div>
                <div className="flex items-start space-x-3">
                  <div className="w-2 h-2 bg-fuchsia-600 rounded-full mt-2"></div>
                  <div>
                    <h4 className="font-medium mb-1">Multi-Platform Ready</h4>
                    <p className="text-gray-600">Expand your reach on YouTube, TikTok, Spotify, and beyond with optimized translations.</p>
                  </div>
                </div>
              </div>
            </div>
            
            <div className="relative">
              <div className="aspect-video w-full rounded-xl bg-white shadow-lg border border-gray-200 overflow-hidden">
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
          </div>
        </div>
      </section>

      {/* Use Cases Section */}
      <section id="use-cases" className="py-20 bg-gray-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-16">
            <h2 className="text-3xl md:text-4xl mb-4 font-bold">
              Grow Your
              <br />
              <span className="text-fuchsia-600">Global Presence</span>
            </h2>
          </div>
          
          <div className="grid md:grid-cols-3 gap-8">
            <Card className="p-6 border-0 bg-white shadow-sm">
              <div className="w-12 h-12 bg-fuchsia-100 rounded-lg flex items-center justify-center mb-4">
                <Youtube className="w-6 h-6 text-fuchsia-600" />
              </div>
              <h3 className="font-semibold mb-3">Dominate Social Platforms</h3>
              <p className="text-gray-600 mb-4">
                Optimize for YouTube, TikTok, Instagram and more with platform-specific translations.
              </p>
              <Badge variant="secondary">Enterprise Ready</Badge>
            </Card>
            
            <Card className="p-6 border-0 bg-white shadow-sm">
              <div className="w-12 h-12 bg-fuchsia-100 rounded-lg flex items-center justify-center mb-4">
                <Globe className="w-6 h-6 text-fuchsia-600" />
              </div>
              <h3 className="font-semibold mb-3">Multiple Languages</h3>
              <p className="text-gray-600 mb-4">
                Translate into dozens of languages with a single click and minimal effort.
              </p>
              <Badge variant="secondary">40+ Languages</Badge>
            </Card>
            
            <Card className="p-6 border-0 bg-white shadow-sm">
              <div className="w-12 h-12 bg-fuchsia-100 rounded-lg flex items-center justify-center mb-4">
                <Music className="w-6 h-6 text-fuchsia-600" />
              </div>
              <h3 className="font-semibold mb-3">Authentic Audience Connection</h3>
              <p className="text-gray-600 mb-4">
                Connect with international audiences in their native language, building loyalty worldwide.
              </p>
              <Badge variant="secondary">Real-time</Badge>
            </Card>
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-20">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
          <h2 className="text-3xl md:text-4xl mb-6 font-bold">
            Ready to Transform
            <br />
            <span className="text-fuchsia-600">How You Communicate?</span>
          </h2>
          <p className="text-xl text-gray-600 mb-8">
            Join the future of authentic, expressive communication across all languages.
          </p>
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <Link to="/creator-studio">
              <Button size="lg" className="group bg-fuchsia-600 hover:bg-fuchsia-700">
                Start Free Trial
                <ArrowRight className="w-4 h-4 ml-2 group-hover:translate-x-1 transition-transform" />
              </Button>
            </Link>
            <Button variant="outline" size="lg">
              Schedule Demo
            </Button>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="border-t border-gray-200 bg-gray-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
          <div className="grid md:grid-cols-4 gap-8">
            <div className="col-span-2">
              <div className="flex items-center space-x-2 mb-4">
                <div className="w-8 h-8 bg-fuchsia-600 rounded-lg flex items-center justify-center">
                  <AudioWaveform className="w-5 h-5 text-white" />
                </div>
                <span className="text-xl font-semibold">Magenta AI</span>
              </div>
              <p className="text-gray-600 mb-4 max-w-md">
                Pioneering expressive speech-to-speech translation that preserves the full spectrum of human communication.
              </p>
              <div className="flex space-x-4">
                <Badge variant="outline">40+ Languages</Badge>
                <Badge variant="outline">Voice Preservation</Badge>
                <Badge variant="outline">Real-time</Badge>
              </div>
            </div>
            
            <div>
              <h4 className="font-medium mb-4">Product</h4>
              <ul className="space-y-2 text-gray-600">
                <li><a href="#" className="hover:text-gray-900 transition-colors">Features</a></li>
                <li><a href="#" className="hover:text-gray-900 transition-colors">API</a></li>
                <li><Link to="/pricing" className="hover:text-gray-900 transition-colors">Pricing</Link></li>
                <li><a href="#" className="hover:text-gray-900 transition-colors">Documentation</a></li>
              </ul>
            </div>
            
            <div>
              <h4 className="font-medium mb-4">Company</h4>
              <ul className="space-y-2 text-gray-600">
                <li><a href="#" className="hover:text-gray-900 transition-colors">About</a></li>
                <li><a href="#" className="hover:text-gray-900 transition-colors">Blog</a></li>
                <li><a href="#" className="hover:text-gray-900 transition-colors">Careers</a></li>
                <li><a href="#" className="hover:text-gray-900 transition-colors">Contact</a></li>
              </ul>
            </div>
          </div>
          
          <div className="border-t border-gray-200 mt-12 pt-8 flex flex-col md:flex-row justify-between items-center">
            <p className="text-gray-600">© {new Date().getFullYear()} Magenta AI. All rights reserved.</p>
            <div className="flex space-x-6 mt-4 md:mt-0">
              <a href="#" className="text-gray-600 hover:text-gray-900 transition-colors">Privacy</a>
              <a href="#" className="text-gray-600 hover:text-gray-900 transition-colors">Terms</a>
              <a href="#" className="text-gray-600 hover:text-gray-900 transition-colors">Security</a>
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