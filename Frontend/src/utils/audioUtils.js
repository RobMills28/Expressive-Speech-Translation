export const validateAudioBlob = async (blob) => {
    if (blob.size === 0) {
      throw new Error('Received empty audio data');
    }
  
    const url = URL.createObjectURL(blob);
    
    try {
      await new Promise((resolve, reject) => {
        const audio = new Audio();
        const timeout = setTimeout(() => {
          audio.src = '';
          reject(new Error('Audio loading timed out'));
        }, 5000);
  
        audio.oncanplaythrough = () => {
          clearTimeout(timeout);
          resolve();
        };
  
        audio.onerror = () => {
          clearTimeout(timeout);
          reject(new Error('Audio format not supported'));
        };
  
        audio.src = url;
      });
      
      return url;
    } catch (error) {
      URL.revokeObjectURL(url);
      throw error;
    }
  };
  
  export const validateAudioFile = (file) => {
    const validExtensions = ['.mp3', '.wav', '.ogg', '.m4a'];
    const validMimeTypes = [
      'audio/mp3', 'audio/mpeg', 'audio/wav', 
      'audio/wave', 'audio/x-wav', 'audio/ogg', 
      'audio/x-m4a', 'audio/mp4', 'audio/aac'
    ];
  
    const fileExtension = file.name.toLowerCase().slice(file.name.lastIndexOf('.'));
    
    if (!validExtensions.includes(fileExtension)) {
      throw new Error(`Invalid file extension. Supported: ${validExtensions.join(', ')}`);
    }
  
    if (!validMimeTypes.includes(file.type) && file.type !== '') {
      console.warn(`Warning: Unexpected MIME type ${file.type}`);
    }
  
    return true;
  };