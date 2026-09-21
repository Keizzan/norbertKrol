<script setup>
import { onMounted } from 'vue'
import AOS from 'aos'
import HeroSection from './components/HeroSection.vue'
import AboutSection from './components/AboutSection.vue'
import SkillsSection from './components/SkillsSection.vue'
import ResumeSection from './components/ResumeSection.vue'
import PortfolioSection from './components/PortfolioSection.vue'
import ContactSection from './components/ContactSection.vue'
import NavBar from './components/NavBar.vue'
import FooterSection from './components/FooterSection.vue'


onMounted(() => {
  AOS.init({ duration: 600, easing: 'ease-in-out', once: true, mirror: false })
})
</script>

<template>
  <main class="main">
    <NavBar />
    <HeroSection />
    <AboutSection />
    <SkillsSection />
    <ResumeSection />
    <PortfolioSection />
    <ContactSection />
  </main>
  <FooterSection />
</template>

<style scoped></style>