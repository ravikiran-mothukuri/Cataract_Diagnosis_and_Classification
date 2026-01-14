document.addEventListener("DOMContentLoaded", function () {
  setTimeout(function () {
      let flashMessages = document.querySelectorAll(".flash-message");
      flashMessages.forEach(function (message) {
          message.style.transition = "opacity 0.5s ease";
          message.style.opacity = "0";
          setTimeout(() => message.remove(), 500); // Removes after fade out
      });
  }, 3000); // 3 seconds delay
});