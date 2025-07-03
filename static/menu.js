document.addEventListener('DOMContentLoaded', function() {
    const menuToggle = document.querySelector('.menu-toggle');
    const sidebar = document.querySelector('.sidebar');
    const mainContent = document.querySelector('.main-content');
    const currentPath = window.location.pathname;

    // Define menu items
    const menuItems = [
        { href: '/', text: 'Home Page' },
        { href: '/chat', text: 'Instant Playlist' },
        { href: '/similarity', text: 'Playlist from Similar Song' }
    ];

    // Find the navigation list
    const navList = document.querySelector('.sidebar-nav');

    // Generate menu items
    menuItems.forEach(item => {
        const listItem = document.createElement('li');
        const link = document.createElement('a');
        link.href = item.href;
        link.textContent = item.text;

        // Check for active page, handling trailing slashes and the root path correctly.
        const linkPath = item.href;
        if (linkPath === '/') {
            // Special case for the home page to match '/' or '/index.html'
            if (currentPath === '/' || currentPath.endsWith('/index.html')) {
                link.classList.add('active');
            }
        } else {
            // For other pages, check for an exact match or a match with a trailing slash.
            if (currentPath === linkPath || currentPath === linkPath + '/') {
                link.classList.add('active');
            }
        }

        listItem.appendChild(link);
        navList.appendChild(listItem);
    });

    // Function to toggle the menu
    const toggleMenu = () => {
        sidebar.classList.toggle('closed');
        mainContent.classList.toggle('full-width');
    };

    // Event listener for the menu toggle button
    if (menuToggle) {
        menuToggle.addEventListener('click', toggleMenu);
    }
});
