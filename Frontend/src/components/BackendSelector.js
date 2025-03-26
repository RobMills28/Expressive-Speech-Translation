import React, { useState, useEffect } from 'react';

const BackendSelector = ({ onBackendChange, className }) => {
  const [backends, setBackends] = useState([]);
  const [selectedBackend, setSelectedBackend] = useState('seamless');
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Fetch available backends on component mount
    const fetchBackends = async () => {
      try {
        setLoading(true);
        const response = await fetch('http://localhost:5001/available-backends');
        if (response.ok) {
          const data = await response.json();
          setBackends(Object.keys(data.backends || {}));
          setSelectedBackend(data.default || 'seamless');
        } else {
          console.error('Failed to fetch backends');
          // Default fallback if API fails
          setBackends(['seamless', 'espnet']);
          setSelectedBackend('seamless');
        }
      } catch (error) {
        console.error('Error fetching backends:', error);
        // Default fallback if API fails
        setBackends(['seamless', 'espnet']);
        setSelectedBackend('seamless');
      } finally {
        setLoading(false);
      }
    };

    fetchBackends();
  }, []);

  const handleBackendChange = (e) => {
    const value = e.target.value;
    setSelectedBackend(value);
    if (onBackendChange) {
      onBackendChange(value);
    }
  };

  return (
    <div className={className}>
      <label className="block text-sm font-medium text-gray-700 mb-1">
        Translation Engine
      </label>
      <select
        value={selectedBackend}
        onChange={handleBackendChange}
        disabled={loading}
        className="mt-1 block w-full pl-3 pr-10 py-2 text-base border-gray-300 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm rounded-md"
      >
        {loading ? (
          <option>Loading...</option>
        ) : (
          backends.map(backend => (
            <option key={backend} value={backend}>
              {backend === 'seamless' ? 'Seamless (Default)' : 
               backend === 'espnet' ? 'ESPnet (Experimental)' : 
               backend}
            </option>
          ))
        )}
      </select>
      {selectedBackend === 'espnet' && (
        <p className="mt-1 text-xs text-amber-600">
          ESPnet is experimental and currently supports limited languages.
        </p>
      )}
    </div>
  );
};

export default BackendSelector;