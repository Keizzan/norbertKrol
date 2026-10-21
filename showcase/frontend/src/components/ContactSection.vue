<script setup>
import { reactive, ref } from 'vue'

// Default form variables
const form = reactive({
  name: '',
  email: '',
  subject: '',
  message: '',
})

// Error message
const errorMessage = ref('')

// Form submit function
async function contactSubmit() {
  errorMessage.value = ''

  try {
    console.log({ ...form })
    // Sent request to backend
    const response = await fetch('https://norbertkrol-backend.onrender.com/submit', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(form),
    })
    // If response negative, throw an error
    if (!response.ok) throw new Error('Failed to send message')

    // Reset form
    form.name = ''
    form.email = ''
    form.subject = ''
    form.message = ''
    // If error occurs, it will be caught in the catch block
  } catch (err) {
    errorMessage.value = err.message
  }
}
</script>

<template>
  <section id="contact" class="contact section">
    <div class="container section-title" data-aos="fade-up mx-auto">
      <h2>Contact</h2>
    </div>

    <div class="container" data-aos="fade" data-aos-delay="100">
      <div class="row gy-4">
        <div class="col-lg-8 mx-auto">
          <form class="php-email-form" data-aos="fade-up" data-aos-delay="200" @submit.prevent="contactSubmit">
            <div class="row gy-4">
              <div class="col-md-6">
                <input v-model="form.name" type="text" name="name" class="form-control" placeholder="Your Name"
                  required />
              </div>

              <div class="col-md-6">
                <input v-model="form.email" type="email" name="email" class="form-control" placeholder="Your Email"
                  required />
              </div>

              <div class="col-md-12">
                <input v-model="form.subject" type="text" name="subject" class="form-control" placeholder="Subject"
                  required />
              </div>

              <div class="col-md-12">
                <textarea v-model="form.message" name="message" class="form-control" rows="6" placeholder="Message"
                  required></textarea>
              </div>

              <div class="col-md-12 text-center">
                <div v-if="errorMessage" class="error-message">{{ errorMessage }}</div>

                <button type="submit">Send Message</button>
              </div>
            </div>
          </form>
        </div>
      </div>
    </div>
  </section>
</template>