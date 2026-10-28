import { useEffect, useState } from 'react'

export type Theme = 'dark' | 'light'

/** Kept in sync with the inline guard script in index.html. */
const STORAGE_KEY = 'kanal-theme'
const DEFAULT_THEME: Theme = 'dark'

function readStoredTheme(): Theme {
  try {
    const stored = window.localStorage.getItem(STORAGE_KEY)
    return stored === 'light' || stored === 'dark' ? stored : DEFAULT_THEME
  } catch {
    return DEFAULT_THEME
  }
}

/**
 * Dark theme is the default; the choice is remembered in localStorage and
 * exposed on <html data-theme="...">, which is what the CSS tokens react to.
 */
export function useTheme() {
  const [theme, setTheme] = useState<Theme>(readStoredTheme)

  useEffect(() => {
    document.documentElement.dataset.theme = theme
    try {
      window.localStorage.setItem(STORAGE_KEY, theme)
    } catch {
      /* private mode: the theme simply will not persist */
    }
  }, [theme])

  return { theme, setTheme }
}
