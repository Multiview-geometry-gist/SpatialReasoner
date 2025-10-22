# Fork를 독립적인 Repository로 전환하기

현재 상황: `Multiview-geometry-gist/SpatialReasoner`는 원본 repo의 fork입니다.
목표: Fork 관계를 끊고 완전히 독립적인 repository로 만들기

## 방법 1: 새 Repository 생성 (추천)

### Step 1: GitHub에서 새 Repository 생성

1. https://github.com/new 접속
2. Repository 이름: `SpatialReasoner-QuaternionResearch` (또는 원하는 이름)
3. **Private/Public 선택**
4. ✅ **"Add a README file" 체크 해제**
5. ✅ **".gitignore" 선택 안 함**
6. ✅ **"Choose a license" 선택 안 함**
7. "Create repository" 클릭

### Step 2: 로컬에서 Remote 변경

```bash
cd /home/user/SpatialReasoner

# 현재 origin 제거
git remote remove origin

# 새 repository를 origin으로 추가
git remote add origin https://github.com/Multiview-geometry-gist/SpatialReasoner-QuaternionResearch.git

# 확인
git remote -v
```

### Step 3: 모든 내용 Push

```bash
# 현재 브랜치 push
git push -u origin claude/spatial-reasoning-research-011CUMFd8XbCcgNLDnUZaPGk

# 또는 main branch로 push하고 싶다면
git checkout -b main
git push -u origin main
```

---

## 방법 2: 현재 Fork에서 Upstream 제거 (간단)

Fork 관계는 유지하되, 원본 repo와의 동기화를 완전히 끊는 방법:

```bash
cd /home/user/SpatialReasoner

# upstream이 설정되어 있다면 제거
git remote remove upstream

# 이제 origin(your fork)만 남음
git remote -v
```

이렇게 하면 기술적으로는 여전히 fork지만, 실질적으로는 독립적으로 작업 가능합니다.

**단점**: GitHub UI에서 여전히 "forked from..." 표시가 남습니다.

---

## 방법 3: GitHub Support에 Fork 관계 해제 요청

GitHub Support에 연락하여 fork 관계를 완전히 끊어달라고 요청할 수 있습니다:

1. https://support.github.com/contact 접속
2. "Repository" → "Detach fork" 선택
3. Repository URL과 이유 설명

**장점**: Fork 관계가 완전히 사라짐
**단점**: 처리에 1-3일 소요

---

## 추천 방안

현재 상황에서는 **방법 1 (새 Repository 생성)**을 추천합니다:

### 이유:
1. ✅ **완전한 독립성**: Fork 관계가 전혀 없음
2. ✅ **이름 변경 가능**: 연구 목적에 맞게 새 이름 설정
3. ✅ **즉시 가능**: GitHub Support 대기 불필요
4. ✅ **깔끔한 히스토리**: 필요한 커밋만 가져올 수 있음

### 새 Repository 이름 제안:
- `SpatialReasoner-Quaternion`
- `SpatialReasoner-MultiView`
- `3D-Spatial-Reasoning-VLM`
- `Quaternion-Spatial-VLM`

---

## 현재 작업 내용 백업

새 repo를 만들기 전에 현재 작업을 안전하게 보관:

```bash
# 현재 디렉토리 전체 백업
cd /home/user
tar -czf SpatialReasoner_backup_$(date +%Y%m%d).tar.gz SpatialReasoner/

# 또는 Git bundle로 저장
cd SpatialReasoner
git bundle create ../spatialreasoner.bundle --all
```

---

## 다음 단계

1. 위 방법 중 하나 선택
2. 새 repository URL 알려주시면 제가 push 도와드리겠습니다
3. examples/ 폴더와 모든 변경사항 업로드
