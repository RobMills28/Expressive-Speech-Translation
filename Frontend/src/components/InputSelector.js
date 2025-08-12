// src/components/InputSelector.js
import React, { useState } from 'react';
import { Upload, Link } from 'lucide-react';
import { Button } from './ui/button';
import { Input } from './ui/input';

const InputSelector = ({ onFileChange, onUrlSubmit, contentType }) => {
  const [inputType, setInputType] = useState('upload');
  const [url, setUrl] = useState('');

  const acceptType = contentType === 'audio' ? 'audio/*,video/*' : 'video/*';
  const uploadText = contentType === 'audio' ? 'Upload Audio or Video File' : 'Upload Video';

  const handleUrlKeyDown = (e) => {
    if (e.key === 'Enter') {
      onUrlSubmit(url);
    }
  };

  return (
    <div className="w-full h-full flex flex-col items-center justify-center p-4 bg-gray-50 rounded-lg border-2 border-dashed border-gray-200">
      <div className="flex space-x-2 mb-4">
        <Button
          variant={inputType === 'upload' ? 'secondary' : 'ghost'}
          onClick={() => setInputType('upload')}
          className="rounded-full"
        >
          <Upload className="h-4 w-4 mr-2" />
          Upload File
        </Button>
        <Button
          variant={inputType === 'url' ? 'secondary' : 'ghost'}
          onClick={() => setInputType('url')}
          className="rounded-full"
        >
          <Link className="h-4 w-4 mr-2" />
          Paste URL
        </Button>
      </div>

      {inputType === 'upload' ? (
        <label className="cursor-pointer w-full flex-grow flex flex-col items-center justify-center text-center">
          <Upload className="w-10 h-10 text-fuchsia-600 mb-2" />
          <span className="text-gray-600 text-sm font-medium">{uploadText}</span>
          <p className="text-xs text-gray-500 mt-1">or drag and drop</p>
          <input type="file" className="hidden" accept={acceptType} onChange={onFileChange} />
        </label>
      ) : (
        <div className="w-full flex flex-col items-center justify-center text-center px-4">
           <p className="text-sm text-gray-600 mb-2">Paste a direct link to an audio or video file.</p>
           <div className="flex w-full max-w-sm space-x-2">
            <Input
              type="url"
              placeholder="https://.../example.mp4"
              value={url}
              onChange={(e) => setUrl(e.target.value)}
              onKeyDown={handleUrlKeyDown}
              className="flex-grow"
            />
            <Button type="button" onClick={() => onUrlSubmit(url)} className="bg-fuchsia-600 hover:bg-fuchsia-700">Go</Button>
          </div>
        </div>
      )}
    </div>
  );
};

export default InputSelector;