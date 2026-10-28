import type { Theme } from '../hooks/useTheme'

interface ThemeToggleProps {
  theme: Theme
  onSelect: (theme: Theme) => void
}

const OPTIONS: Array<{ value: Theme; label: string }> = [
  { value: 'dark', label: 'تاریک' },
  { value: 'light', label: 'روشن' },
]

/** A compact two-state switch; the two themes share one set of tokens. */
function ThemeToggle({ theme, onSelect }: ThemeToggleProps) {
  return (
    <div className="theme-toggle" role="group" aria-label="انتخاب تم">
      {OPTIONS.map((option) => (
        <button
          key={option.value}
          type="button"
          className="theme-toggle__option"
          aria-pressed={theme === option.value}
          onClick={() => onSelect(option.value)}
        >
          {option.label}
        </button>
      ))}
    </div>
  )
}

export default ThemeToggle
