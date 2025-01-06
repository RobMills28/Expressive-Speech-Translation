import React, { useState, useRef } from 'react';
import { Card, CardContent } from "./ui/card";
import { Button } from "./ui/button";
import { PlayCircle, Heart, Upload } from 'lucide-react';
import { Alert, AlertDescription } from "./ui/alert";

const PodcastPage = () => {
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState('');
  const fileInputRef = useRef(null);
  
  // Initialize with some sample podcasts but allow for new ones
  const [podcasts, setPodcasts] = useState([
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
  ]);

  const handleUploadClick = () => {
    fileInputRef.current?.click();
  };

  const formatDuration = (seconds) => {
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = Math.floor(seconds % 60);
    return `${String(minutes).padStart(2, '0')}:${String(remainingSeconds).padStart(2, '0')}`;
  };

  const handleFileSelect = async (event) => {
    const file = event.target.files?.[0];
    if (!file) return;

    try {
      setUploading(true);
      setError('');

      const formData = new FormData();
      formData.append('file', file);
      formData.append('title', file.name.replace(/\.[^/.]+$/, ''));
      formData.append('date', new Date().toISOString());

      const response = await fetch('http://localhost:5001/upload_podcast', {
        method: 'POST',
        body: formData,
        credentials: 'include'
      });

      if (!response.ok) {
        throw new Error(await response.text());
      }

      const newPodcast = await response.json();
      
      // Add the new podcast to the list with proper formatting
      setPodcasts(prevPodcasts => [{
        id: newPodcast.id,
        title: newPodcast.title,
        episode: `EP ${newPodcast.episode}`,
        duration: newPodcast.duration,
        thumbnail: "/api/placeholder/192/192", // Use placeholder for now
        languages: ["🇺🇸 EN"], // Start with English, more languages can be added after translation
        date: new Date(newPodcast.date).toLocaleDateString('en-US', {
          month: 'short',
          day: 'numeric',
          year: 'numeric'
        })
      }, ...prevPodcasts]);

      // Reset file input
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
    } catch (error) {
      console.error('Upload failed:', error);
      setError('Failed to upload podcast. Please try again.');
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-700 via-fuchsia-500 to-pink-500">
      <div className="container mx-auto p-6 space-y-6">
        {/* Upload Section */}
        <Card className="bg-white/90 backdrop-blur-md shadow-xl">
          <CardContent className="p-6">
            <div className="space-y-2">
              <h2 className="text-xl font-semibold text-gray-800">Upload Your Podcast</h2>
              <p className="text-sm text-gray-600">Translate your content into multiple languages</p>
              
              <input
                ref={fileInputRef}
                type="file"
                accept="audio/*"
                onChange={handleFileSelect}
                className="hidden"
              />
              
              <Button
                onClick={handleUploadClick}
                disabled={uploading}
                className="w-full bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-700 hover:to-pink-700 text-white"
              >
                {uploading ? (
                  <div className="flex items-center justify-center gap-2">
                    <div className="animate-spin h-4 w-4 border-2 border-white border-t-transparent rounded-full" />
                    <span>Uploading...</span>
                  </div>
                ) : (
                  <div className="flex items-center justify-center gap-2">
                    <Upload className="h-4 w-4" />
                    <span>Upload Podcast</span>
                  </div>
                )}
              </Button>

              {error && (
                <Alert variant="destructive" className="mt-2">
                  <AlertDescription>{error}</AlertDescription>
                </Alert>
              )}
            </div>
          </CardContent>
        </Card>

        {/* Recent Episodes */}
        <Card className="bg-white/90 backdrop-blur-md shadow-xl">
          <CardContent className="p-6">
            <div className="space-y-4">
              <div>
                <h2 className="text-xl font-semibold text-gray-800">Recent Episodes</h2>
                <p className="text-sm text-gray-600">Browse and translate your podcast episodes</p>
              </div>

              <div className="space-y-4">
                {podcasts.map((podcast) => (
                  <div 
                    key={podcast.id}
                    className="flex items-center gap-4 p-4 hover:bg-gray-50 rounded-lg transition-colors"
                  >
                    <img 
                      src={podcast.thumbnail}
                      alt={podcast.title}
                      className="w-16 h-16 rounded-lg object-cover"
                    />
                    <div className="flex-1 min-w-0">
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
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
};

export default PodcastPage;