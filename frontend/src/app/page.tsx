'use client'

import { useState, useEffect } from 'react'
import axios from 'axios'

// Streamlit과 동일한 클래스 정의
const CLASS_NAMES = ["bump", "fall-down", "fall-off", "hit", "jam", "no-accident"]
const CLASS_NAMES_KR = ["충돌 사고", "넘어짐 사고", "추락 사고", "타격 사고", "끼임 사고", "정상 상황"]

export default function Home() {
  const [file, setFile] = useState<File | null>(null)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<any>(null)
  const [modelStatus, setModelStatus] = useState<any>(null)
  const [isMobile, setIsMobile] = useState(false)

  // 컴포넌트 마운트 시 모델 상태 확인 및 화면 크기 감지
  useEffect(() => {
    checkModelStatus()
    
    // 화면 크기 감지
    const checkScreenSize = () => {
      setIsMobile(window.innerWidth <= 768)
    }
    
    checkScreenSize()
    window.addEventListener('resize', checkScreenSize)
    
    return () => window.removeEventListener('resize', checkScreenSize)
  }, [])

  const checkModelStatus = async () => {
    try {
      const response = await axios.get('/api/health')
      setModelStatus(response.data)
    } catch (error) {
      console.error('Health check failed:', error)
    }
  }

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0])
      setResult(null)
    }
  }

  const handleUpload = async () => {
    if (!file) return

    setLoading(true)
    const formData = new FormData()
    formData.append('file', file)

    try {
      const response = await axios.post('/api/predict', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      })
      setResult(response.data)
    } catch (error) {
      console.error('Upload failed:', error)
      alert('업로드 실패')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div style={{ 
      fontFamily: 'Arial, sans-serif',
      backgroundColor: '#fafafa',
      minHeight: '100vh',
      padding: '0'
    }}>
      {/* Streamlit 스타일 헤더 */}
      <div style={{
        backgroundColor: 'white',
        padding: isMobile ? '1rem 1rem' : '1rem 2rem',
        borderBottom: '1px solid #e6e6e6',
        marginBottom: isMobile ? '1rem' : '2rem'
      }}>
        <h1 style={{ 
          color: '#262730',
          fontSize: isMobile ? '1.8rem' : '2.5rem',
          fontWeight: '600',
          margin: '0',
          display: 'flex',
          alignItems: 'center',
          gap: '0.5rem',
          flexWrap: 'wrap'
        }}>
          🏭 제조업 안전사고 감지 시스템
        </h1>
        <div style={{
          height: '3px',
          background: 'linear-gradient(90deg, #ff6b6b, #4ecdc4)',
          marginTop: '1rem'
        }}></div>
      </div>

      <div style={{ 
        maxWidth: '1200px', 
        margin: '0 auto', 
        padding: isMobile ? '0 1rem' : '0 2rem' 
      }}>
        
        {/* 모델 상태 표시 */}
        {modelStatus && (
          <div style={{
            backgroundColor: modelStatus.model_loaded ? '#d4edda' : '#f8d7da',
            color: modelStatus.model_loaded ? '#155724' : '#721c24',
            padding: '0.75rem 1rem',
            borderRadius: '0.375rem',
            border: `1px solid ${modelStatus.model_loaded ? '#c3e6cb' : '#f5c6cb'}`,
            marginBottom: '1.5rem',
            fontSize: '0.9rem'
          }}>
            {modelStatus.model_loaded ? '✅' : '❌'} 모델 상태: {modelStatus.model_loaded ? '로드 완료' : '로드 실패'} (Device: {modelStatus.device})
          </div>
        )}

        <div style={{ 
          display: 'grid', 
          gridTemplateColumns: isMobile ? '1fr' : '1fr 1fr',
          gap: isMobile ? '1.5rem' : '2rem'
        }}>
          
          {/* 왼쪽: 비디오 업로드 */}
          <div>
            <div style={{
              backgroundColor: 'white',
              padding: isMobile ? '1rem' : '1.5rem',
              borderRadius: '0.5rem',
              boxShadow: '0 1px 3px rgba(0,0,0,0.1)',
              border: '1px solid #e6e6e6'
            }}>
              <h2 style={{ 
                color: '#262730',
                fontSize: isMobile ? '1.3rem' : '1.5rem',
                fontWeight: '600',
                marginBottom: '1rem',
                display: 'flex',
                alignItems: 'center',
                gap: '0.5rem'
              }}>
                📤 비디오 업로드
              </h2>
              
              <div style={{
                border: '2px dashed #cccccc',
                borderRadius: '0.5rem',
                padding: isMobile ? '1.5rem' : '2rem',
                textAlign: 'center',
                backgroundColor: '#fafafa'
              }}>
                <input
                  type="file"
                  accept="video/*"
                  onChange={handleFileChange}
                  style={{
                    marginBottom: '1rem',
                    padding: '0.5rem',
                    border: '1px solid #ddd',
                    borderRadius: '0.25rem',
                    width: '100%'
                  }}
                />
                
                <p style={{ color: '#666', fontSize: '0.9rem', margin: '0.5rem 0' }}>
                  지원 형식: MP4, AVI, MOV, MKV, WMV
                </p>
                
                {file && (
                  <div style={{ marginTop: '1rem' }}>
                    <p style={{ color: '#262730', fontWeight: '500', marginBottom: '1rem' }}>
                      선택된 파일: {file.name}
                    </p>
                    <video 
                      src={URL.createObjectURL(file)} 
                      controls 
                      style={{ 
                        width: '100%', 
                        maxHeight: '300px',
                        borderRadius: '0.375rem'
                      }}
                    />
                  </div>
                )}
              </div>

              {file && (
                <button
                  onClick={handleUpload}
                  disabled={loading}
                  style={{
                    width: '100%',
                    padding: '0.75rem 1.5rem',
                    backgroundColor: loading ? '#6c757d' : '#ff4b4b',
                    color: 'white',
                    border: 'none',
                    borderRadius: '0.375rem',
                    fontSize: '1rem',
                    fontWeight: '500',
                    cursor: loading ? 'not-allowed' : 'pointer',
                    marginTop: '1rem',
                    transition: 'background-color 0.2s'
                  }}
                  onMouseOver={(e) => {
                    if (!loading) e.currentTarget.style.backgroundColor = '#e63946'
                  }}
                  onMouseOut={(e) => {
                    if (!loading) e.currentTarget.style.backgroundColor = '#ff4b4b'
                  }}
                >
                  {loading ? '비디오 분석 중...' : '🔍 안전사고 분석 시작'}
                </button>
              )}
            </div>
          </div>

          {/* 오른쪽: 분석 결과 */}
          <div>
            {result ? (
              <div style={{
                backgroundColor: 'white',
                padding: isMobile ? '1rem' : '1.5rem',
                borderRadius: '0.5rem',
                boxShadow: '0 1px 3px rgba(0,0,0,0.1)',
                border: '1px solid #e6e6e6'
              }}>
                <h2 style={{ 
                  color: '#262730',
                  fontSize: isMobile ? '1.3rem' : '1.5rem',
                  fontWeight: '600',
                  marginBottom: '1rem',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '0.5rem'
                }}>
                  📊 분석 결과
                </h2>

                {/* 예측 결과 */}
                <div style={{
                  backgroundColor: result.prediction.is_accident ? '#f8d7da' : '#d4edda',
                  color: result.prediction.is_accident ? '#721c24' : '#155724',
                  padding: '1rem',
                  borderRadius: '0.375rem',
                  border: `1px solid ${result.prediction.is_accident ? '#f5c6cb' : '#c3e6cb'}`,
                  marginBottom: '1.5rem',
                  textAlign: 'center'
                }}>
                  <div style={{ fontSize: '1.5rem', fontWeight: '600', marginBottom: '0.5rem' }}>
                    {result.prediction.is_accident ? '⚠️ ' + result.prediction.class_name_kr + ' 감지!' : '✅ ' + result.prediction.class_name_kr}
                  </div>
                  <div style={{ fontSize: '1.2rem', fontWeight: '500' }}>
                    신뢰도: {(result.prediction.confidence * 100).toFixed(2)}%
                  </div>
                  <div style={{ fontSize: '0.9rem', marginTop: '0.5rem', opacity: 0.8 }}>
                    영문 클래스: {result.prediction.class_name}
                  </div>
                </div>

                {/* 클래스별 확률 */}
                <div>
                  <h3 style={{ 
                    color: '#262730',
                    fontSize: '1.2rem',
                    fontWeight: '600',
                    marginBottom: '1rem'
                  }}>
                    📈 클래스별 확률
                  </h3>
                  
                  <div style={{ marginBottom: '1rem' }}>
                    {Object.entries(result.probabilities).map(([className, prob]: [string, any], index) => {
                      const probValues = Object.values(result.probabilities) as number[]
                      const isHighest = prob === Math.max(...probValues)
                      return (
                        <div key={className} style={{ marginBottom: '0.5rem' }}>
                          <div style={{ 
                            display: 'flex', 
                            justifyContent: 'space-between',
                            alignItems: 'center',
                            marginBottom: '0.25rem',
                            fontSize: '0.9rem',
                            fontWeight: isHighest ? '600' : '400',
                            color: isHighest ? '#262730' : '#666'
                          }}>
                            <span>{className}</span>
                            <span>{(prob * 100).toFixed(2)}%</span>
                          </div>
                          <div style={{
                            width: '100%',
                            height: '8px',
                            backgroundColor: '#e9ecef',
                            borderRadius: '4px',
                            overflow: 'hidden'
                          }}>
                            <div style={{
                              width: `${prob * 100}%`,
                              height: '100%',
                              backgroundColor: isHighest ? '#ff4b4b' : '#6c757d',
                              transition: 'width 0.3s ease'
                            }}></div>
                          </div>
                        </div>
                      )
                    })}
                  </div>

                  {/* 확률 테이블 */}
                  <div style={{
                    border: '1px solid #e6e6e6',
                    borderRadius: '0.375rem',
                    overflow: 'hidden'
                  }}>
                    <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                      <thead>
                        <tr style={{ backgroundColor: '#f8f9fa' }}>
                          <th style={{ padding: '0.75rem', textAlign: 'left', borderBottom: '1px solid #e6e6e6', fontSize: '0.9rem', fontWeight: '600' }}>클래스</th>
                          <th style={{ padding: '0.75rem', textAlign: 'left', borderBottom: '1px solid #e6e6e6', fontSize: '0.9rem', fontWeight: '600' }}>영문</th>
                          <th style={{ padding: '0.75rem', textAlign: 'right', borderBottom: '1px solid #e6e6e6', fontSize: '0.9rem', fontWeight: '600' }}>확률</th>
                        </tr>
                      </thead>
                      <tbody>
                        {CLASS_NAMES_KR.map((classKr, index) => {
                          const classEn = CLASS_NAMES[index]
                          const prob = result.probabilities[classKr] || 0
                          const probValues = Object.values(result.probabilities) as number[]
                          const isHighest = prob === Math.max(...probValues)
                          return (
                            <tr key={classKr} style={{ 
                              backgroundColor: isHighest ? '#fff3cd' : 'white',
                              borderBottom: index < CLASS_NAMES_KR.length - 1 ? '1px solid #e6e6e6' : 'none'
                            }}>
                              <td style={{ padding: '0.75rem', fontSize: '0.9rem', fontWeight: isHighest ? '600' : '400' }}>{classKr}</td>
                              <td style={{ padding: '0.75rem', fontSize: '0.9rem', color: '#666', fontFamily: 'monospace' }}>{classEn}</td>
                              <td style={{ padding: '0.75rem', textAlign: 'right', fontSize: '0.9rem', fontWeight: isHighest ? '600' : '400' }}>
                                {(prob * 100).toFixed(2)}%
                              </td>
                            </tr>
                          )
                        })}
                      </tbody>
                    </table>
                  </div>
                </div>
              </div>
            ) : (
              <div style={{
                backgroundColor: 'white',
                padding: '2rem',
                borderRadius: '0.5rem',
                boxShadow: '0 1px 3px rgba(0,0,0,0.1)',
                border: '1px solid #e6e6e6',
                textAlign: 'center',
                color: '#666'
              }}>
                <div style={{ fontSize: '3rem', marginBottom: '1rem' }}>📊</div>
                <p>비디오를 업로드하고 분석을 시작하면<br/>결과가 여기에 표시됩니다</p>
              </div>
            )}
          </div>
        </div>

        {/* 하단 시스템 정보 */}
        <div style={{
          backgroundColor: 'white',
          padding: '1.5rem',
          borderRadius: '0.5rem',
          boxShadow: '0 1px 3px rgba(0,0,0,0.1)',
          border: '1px solid #e6e6e6',
          marginTop: '2rem'
        }}>
          <details>
            <summary style={{ 
              cursor: 'pointer', 
              fontSize: '1.1rem', 
              fontWeight: '600',
              color: '#262730',
              marginBottom: '1rem'
            }}>
              ℹ️ 시스템 정보
            </summary>
            <div style={{ paddingLeft: '1rem', color: '#666', lineHeight: '1.6' }}>
              <p><strong>모델 정보:</strong></p>
              <ul style={{ marginLeft: '1rem' }}>
                <li>아키텍처: CNN-LSTM</li>
                <li>백본: ResNet-101 (사전학습)</li>
                <li>입력 형태: 16프레임 × 224×224 RGB</li>
                <li>클래스 수: 6개</li>
                <li>디바이스: {modelStatus?.device || 'Unknown'}</li>
              </ul>
              
              <p style={{ marginTop: '1rem' }}><strong>지원 사고 유형:</strong></p>
              <ul style={{ marginLeft: '1rem' }}>
                <li>충돌 사고 (bump)</li>
                <li>넘어짐 사고 (fall-down)</li>
                <li>추락 사고 (fall-off)</li>
                <li>타격 사고 (hit)</li>
                <li>끼임 사고 (jam)</li>
                <li>정상 상황 (no-accident)</li>
              </ul>
            </div>
          </details>
        </div>
      </div>
    </div>
  )
}
