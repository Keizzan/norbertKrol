<script setup>
import { onMounted, onUnmounted, ref } from 'vue'

const links = [
  { hash: '#hero', label: 'Home', icon: 'bi-house' },
  { hash: '#about', label: 'About', icon: 'bi-person' },
  { hash: '#skills', label: 'Skills', icon: 'bi-tools' },
  { hash: '#resume', label: 'CV', icon: 'bi-file-earmark-text' },
  { hash: '#portfolio', label: 'Portfolio', icon: 'bi-images' },
  { hash: '#contact', label: 'Contact', icon: 'bi-envelope' },
]

const activeHash = ref('#hero')
const mobileNavOpen = ref(false)

function toggleMobileNav() {
  mobileNavOpen.value = !mobileNavOpen.value
}

function closeMobileNav() {
  mobileNavOpen.value = false
}

function updateActiveLink() {
  const position = window.scrollY + 200

  for (const link of links) {
    const section = document.querySelector(link.hash)
    if (!section) continue

    const top = section.offsetTop
    const bottom = top + section.offsetHeight

    if (position >= top && position <= bottom) {
      activeHash.value = link.hash
      return
    }
  }
}

onMounted(() => {
  updateActiveLink()
  window.addEventListener('scroll', updateActiveLink)
})

onUnmounted(() => {
  window.removeEventListener('scroll', updateActiveLink)
})
</script>

<template>
  <header id="header" class="header d-flex flex-column justify-content-center" :class="{ 'header-show': mobileNavOpen }">
    <i
      class="header-toggle d-xl-none bi"
      :class="mobileNavOpen ? 'bi-x' : 'bi-list'"
      @click="toggleMobileNav"
    ></i>

    <nav id="navmenu" class="navmenu">
      <ul>
        <li v-for="link in links" :key="link.hash">
          <a
            :href="link.hash"
            :class="{ active: activeHash === link.hash }"
            @click="closeMobileNav"
          >
            <i class="bi navicon" :class="link.icon"></i>
            <span>{{ link.label }}</span>
          </a>
        </li>
      </ul>
    </nav>
  </header>
</template>