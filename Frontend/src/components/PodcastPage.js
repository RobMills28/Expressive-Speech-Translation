import React, { useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "./ui/card";
import { Button } from "./ui/button";
import { Globe, Upload, PlayCircle, Heart } from 'lucide-react';

const PodcastPage = () => {
  const [selectedPodcast, setSelectedPodcast] = useState(null);

  const samplePodcasts = [
    {
      id: 1,
      title: "The Future of AI",
      episode: "EP 145",
      duration: "45:00",
      thumbnail: "/api/placeholder/192/192",
      languages: ["🇺🇸 EN", "🇪🇸 ES", "🇫🇷 FR"],
      date: "Jan 2, 2025"
    },
    {
      id: 2,
      title: "Web3 and Digital Finance",
      episode: "EP 144",
      duration: "42:15",
      thumbnail: "/api/placeholder/192/192",
      languages: ["🇺🇸 EN", "🇪🇸 ES"],
      date: "Dec 28, 2024"
    }
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-700 via-fuchsia-500 to-pink-500">
      <div className="container mx-auto p-6">
        {/* Upload Section */}
        <Card className="mb-8 bg-white/90 backdrop-blur-md shadow-xl">
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <h2 className="text-xl font-semibold text-gray-800 mb-2">Upload Your Podcast</h2>
                <p className="text-sm text-gray-600">Translate your content into multiple languages</p>
              </div>
              <Button className="bg-purple-600 hover:bg-purple-700">
                <Upload className="w-4 h-4 mr-2" />
                Upload Podcast
              </Button>
            </div>
          </CardContent>
        </Card>

        {/* Featured Podcast */}
        <Card className="mb-8 bg-white/90 backdrop-blur-md shadow-xl">
          <CardContent className="p-6">
            <div className="flex gap-6">
              <div className="w-48 h-48 bg-gray-200 rounded-lg flex items-center justify-center">
                <img 
                  src="/api/placeholder/192/192" 
                  alt="Featured podcast"
                  className="rounded-lg"
                />
              </div>
              <div className="flex-1">
                <div className="flex items-center gap-3 mb-2">
                  <span className="bg-purple-600 text-white px-2 py-1 rounded text-sm">FEATURED</span>
                  <span className="flex items-center gap-1 text-sm text-gray-600">
                    <Globe className="w-4 h-4" />
                    Available in 5 languages
                  </span>
                </div>
                <h2 className="text-2xl font-bold text-gray-800 mb-2">The Future of AI</h2>
                <p className="text-gray-600 mb-4">Latest episode: The Rise of AI Assistants - EP 145</p>
                
                <div className="flex items-center gap-4">
                  <PlayCircle className="w-12 h-12 text-purple-600 hover:text-purple-700 cursor-pointer" />
                  <div className="flex-1">
                    <div className="h-2 bg-gray-200 rounded-full overflow-hidden">
                      <div className="h-full w-1/3 bg-purple-600"></div>
                    </div>
                    <div className="flex justify-between text-sm text-gray-600 mt-1">
                      <span>14:22</span>
                      <span>45:00</span>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Recent Episodes */}
        <Card className="bg-white/90 backdrop-blur-md shadow-xl">
          <CardHeader>
            <CardTitle>Recent Episodes</CardTitle>
            <CardDescription>Browse and translate your podcast episodes</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {samplePodcasts.map((podcast) => (
                <div 
                  key={podcast.id} 
                  className="flex items-center gap-4 p-4 rounded-lg hover:bg-gray-50 transition-colors"
                >
                  <img 
                    src={podcast.thumbnail}
                    alt={podcast.title}
                    className="w-16 h-16 rounded-lg object-cover"
                  />
                  <div className="flex-1">
                    <h3 className="font-semibold text-gray-800">{podcast.title}</h3>
                    <p className="text-sm text-gray-500">{podcast.episode} • {podcast.date}</p>
                    <div className="flex gap-2 mt-1">
                      {podcast.languages.map((lang, index) => (
                        <span 
                          key={index} 
                          className="text-xs bg-gray-100 text-gray-600 px-2 py-1 rounded"
                        >
                          {lang}
                        </span>
                      ))}
                    </div>
                  </div>
                  <div className="flex items-center gap-3">
                    <span className="text-sm text-gray-500">{podcast.duration}</span>
                    <PlayCircle className="w-8 h-8 text-purple-600 hover:text-purple-700 cursor-pointer" />
                    <Heart className="w-6 h-6 text-gray-400 hover:text-pink-500 cursor-pointer" />
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
};

export default PodcastPage;