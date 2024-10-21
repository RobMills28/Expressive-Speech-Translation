/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./src/**/*.{js,jsx,ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        fuchsia: {
          500: '#d946ef',
          600: '#c026d3',
          700: '#a21caf',
        },
        pink: {
          500: '#ec4899',
          600: '#db2777',
        },
        purple: {
          700: '#7e22ce',
        },
      },
    },
  },
  plugins: [],
}