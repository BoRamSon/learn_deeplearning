import type { Metadata } from 'next'
import './globals.css'

export const metadata: Metadata = {
  title: '안전사고 감지',
  description: 'AI 기반 안전사고 감지 시스템',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="ko">
      <head>
        <meta name="viewport" content="width=device-width, initial-scale=1.0" />
      </head>
      <body style={{ margin: 0, fontFamily: 'Arial, sans-serif' }}>
        {children}
      </body>
    </html>
  )
}